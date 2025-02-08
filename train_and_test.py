import torch
import numpy as np
import random
import logging
import argparse
import os
import time
import torch.nn.functional as F
from torchmetrics import JaccardIndex, Precision, Recall, FBetaScore
import torch.utils.data as data
from lib.CPFANet import CPFANet
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from utils.metrics import calculate_metrics
from utils.dataloader import get_loader, TestDataset
from utils.util import clip_gradient, adjust_lr, save_mask, set_random_seed, CalParams


def get_args():
    parser = argparse.ArgumentParser(description="Args for Polyp Segmentation")
    parser.add_argument("--dataset", type=str,
                        default="polyp5", help="DATASETNAME")
    parser.add_argument("--datadir", type=str,
                        default="./data/polypSegDataset", help="Dir of your dataset, like '/data/DATASETNAME'")
    parser.add_argument("--save_root", type=str,
                        default="./result_maps/CPFANet", help="Dir for saving test predictions")
    parser.add_argument("--checkpoint_path", type=str,
                        default="./snapshots/CPFANet", help="Dir for saving checkpoint_path")
    parser.add_argument("--num_workers", type=int,
                        default=16, help="Number of workers for loading the data")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Batch size, default=16")
    parser.add_argument("--augmentation", type=bool,
                        default=False, help="augmentation: use or not")
    parser.add_argument("--image_size", type=int,
                        default=352, help="Image size, default=352")
    parser.add_argument("--epochs", type=int,
                        default=40, help="Max training epochs, default=40")
    parser.add_argument("--lr", type=float,
                        default=1e-4, help="Learning rate (default: 0.0001)")
    parser.add_argument('--clip', type=float,
                        default=0.5, help='Gradient clipping margin')
    parser.add_argument("--gpu", type=str,
                        default="0", help="input visible devices for training (default: cuda:0)")
    parser.add_argument("--lr_decay_epoch", type=int,
                        default=20, help="Number of epochs before decaying learning rate (default: 20)")
    parser.add_argument("--lr_decay_rate", type=float,
                        default=0.1, help="Decay rate for learning rate (default: 0.1)")
    parser.add_argument("--num_classes", type=int,
                        default=2, help="Number of classes for segmentation (default: 2)")
    args = parser.parse_args(args=[])
    return args


def get_log(file_name):
    logger = logging.getLogger('Polyp Segmentation')
    logger.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    fh = logging.FileHandler(file_name, mode='w')
    sh.setLevel(logging.INFO)
    fh.setLevel(logging.INFO) 
    formatter = logging.Formatter('[%(asctime)s]\t%(message)s',datefmt="%Y-%m-%d %H:%M:%S")
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)
    return logger
    
                
class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice_score = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
        return 1 - dice_score.mean()
        

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    
    return (wbce + wiou).mean()


def train(model, trainloader, optimizer, epoch, args):
    model.train()
    # ---- multi-scale training ----
    size_rates = [0.75, 1, 1.25]

    jaccard = JaccardIndex(task="binary", num_classes=args.num_classes).cuda()
    precision = Precision(task="binary", num_classes=args.num_classes, average='macro').cuda()
    recall = Recall(task="binary", num_classes=args.num_classes, average='macro').cuda()
    f2 = FBetaScore(task="binary", average='macro', beta=2.0).cuda()
    running_loss = 0.0
    metrics = {"mDice": 0.0, "mIoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F2": 0.0}
    total_batches = len(trainloader)
    for batch_idx, pack in enumerate(trainloader, start=1):
        batch_metrics = {"mDice": 0.0, "mIoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F2": 0.0}

        for rate in size_rates: 
            optimizer.zero_grad()
            images, gts = pack
            images = Variable(images).cuda()
            gts = Variable(gts).cuda()

            trainsize = int(round(args.image_size*rate/32)*32)
            if rate != 1:
                images = F.upsample(images, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                gts = F.upsample(gts, size=(trainsize, trainsize), mode='bilinear', align_corners=True)
                
            preds,out2,out3,out4,out5 = model(images)
            loss1 = structure_loss(preds, gts)
            loss2 = structure_loss(out2, gts)
            loss3 = structure_loss(out3, gts)
            loss4 = structure_loss(out4, gts)
            loss5 = structure_loss(out5, gts)
            loss = loss1+loss2+loss3+loss4+loss5
            
            running_loss += loss.item()

            loss.backward()
            clip_gradient(optimizer, args.clip)
            optimizer.step()
            
            preds_sigmoid = torch.sigmoid(preds)
            gts = gts.int()
            batch_metrics["mDice"] += 1 - DiceLoss()(preds, gts).item()
            batch_metrics["mIoU"] += jaccard(preds_sigmoid, gts)
            batch_metrics["Precision"] += precision(preds_sigmoid, gts)
            batch_metrics["Recall"] += recall(preds_sigmoid, gts)
            batch_metrics["F2"] += f2(preds_sigmoid, gts)

        for key in batch_metrics:
            metrics[key] += batch_metrics[key] / len(size_rates)

    # Average loss
    avg_loss = running_loss / total_batches
    metrics = {k: v / total_batches for k, v in metrics.items()}

    logger.info(f"\nEpoch [{epoch}], Loss: {avg_loss:.4f}, mDice: {metrics['mDice']:.4f}, mIoU: {metrics['mIoU']:.4f}, "
                f"Precision: {metrics['Precision']:.4f}, Recall: {metrics['Recall']:.4f}, F2: {metrics['F2']:.4f}, "
                f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    os.makedirs(args.checkpoint_path, exist_ok=True)
    if epoch % args.epochs == 0:
        torch.save(model.state_dict(), os.path.join(args.checkpoint_path, f'CPFANet-{epoch}.pth'))
        logger.info(f"[Saving Snapshot:] {args.checkpoint_path} CPFANet-{epoch}.pth")

    return avg_loss, metrics


def test(model, test_root, args, binary_flag=True):
    model.eval()
    dataset_metrics = {data_name: {"mDice": 0.0, "mIoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F2": 0.0,
                                   "S-measure": 0.0, "F-measure": 0.0, "E-measure": 0.0, "MAE": 0.0} 
                       for data_name in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']}
    save_root = args.save_root
    for _data_name in dataset_metrics.keys():
        data_path = os.path.join(test_root, _data_name)
        save_path = os.path.join(save_root, _data_name)

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        test_loader = TestDataset(os.path.join(data_path, 'images/'),
                                 os.path.join(data_path, 'masks/'),
                                 testsize=args.image_size)

        dataset_metric = {key: 0.0 for key in dataset_metrics[_data_name]}
        total_batches = len(test_loader)

        with torch.no_grad():
            for i in range(test_loader.size):
                images, gts, name = test_loader.load_data()
                gts = test_loader.gt_transform(gts).unsqueeze(0).cuda()
                
                preds = model(images)[0]
                
                preds_sigmoid = torch.sigmoid(preds)
                preds_binary = preds_sigmoid > 0.5
                if binary_flag:
                    calc_scores = calculate_metrics(preds_binary, gts)
                else:
                    calc_scores = calculate_metrics(preds_sigmoid, gts)
                dataset_metric["mDice"] += calc_scores[0]
                dataset_metric["mIoU"] += calc_scores[1]
                dataset_metric["S-measure"] += calc_scores[2]
                dataset_metric["E-measure"] += calc_scores[3]
                dataset_metric["F-measure"] += calc_scores[4]
                dataset_metric["MAE"] += calc_scores[5]
                dataset_metric["Precision"] += calc_scores[6]
                dataset_metric["Recall"] += calc_scores[7]
                dataset_metric["F2"] += calc_scores[8]

                pred_name = name if name.endswith('.png') else name + '.png'
                save_pred_path = os.path.join(save_path, pred_name.replace('.jpg', '_pred.png').replace('.png', '_pred.png'))
                save_mask(preds_sigmoid[0].cpu().numpy(), save_pred_path)
                
        dataset_metrics[_data_name] = {k: v / total_batches for k, v in dataset_metric.items()}

        logger.info(f"Dataset: {_data_name} - "
                    f"mDice: {dataset_metrics[_data_name]['mDice']:.4f}, "
                    f"mIoU: {dataset_metrics[_data_name]['mIoU']:.4f}, Precision: {dataset_metrics[_data_name]['Precision']:.4f}, "
                    f"Recall: {dataset_metrics[_data_name]['Recall']:.4f}, F2: {dataset_metrics[_data_name]['F2']:.4f}, "
                    f"S-measure: {dataset_metrics[_data_name]['S-measure']:.4f}, "
                    f"F-measure: {dataset_metrics[_data_name]['F-measure']:.4f}, "
                    f"E-measure: {dataset_metrics[_data_name]['E-measure']:.4f}, MAE: {dataset_metrics[_data_name]['MAE']:.4f}")

    return dataset_metrics


def main():
    #-------Load hyperparameters-------#
    # set_random_seed(66)
    args = get_args()
    log_path = "./logs/dataset{}_CPFANet_bs{}_epochs{}_lr{}_lrdecay{}_{}_{}.log".format(
        args.dataset, 
        args.batch_size, 
        args.epochs, 
        args.lr, 
        args.lr_decay_epoch, 
        args.lr_decay_rate, 
        time.strftime("%Y%m%d_%H%M%S")
    )
    
    global logger
    logger = get_log(log_path)
    logger.info("Polyp Seg: train information on {} dataset\n".format(args.dataset))
    logger.info("The List of Hyperparameters:")
    for k, v in sorted(vars(args).items()):
        logger.info("--{} = {}".format(k, v))
    
    #-------Load model&data-------#
    model = CPFANet().cuda()
    # input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # CalParams(model, input_tensor)
    params = model.parameters()
    optimizer = torch.optim.AdamW(params, args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_epoch, gamma=args.lr_decay_rate)

    train_root = f"{args.datadir}/TrainDataset/images/"
    train_gt_root = f"{args.datadir}/TrainDataset/masks/"    
    train_loader = get_loader(train_root, train_gt_root, batchsize=args.batch_size,
                              trainsize=args.image_size,num_workers=args.num_workers,pin_memory=True)

    #-------Training-------#
    logger.info("\n\n*************************Training*************************")
    for epoch in range(1, args.epochs+1):
        _, metrics = train(model, train_loader, optimizer, epoch, args)
        scheduler.step()
    
    #--------Testing--------#
    logger.info("\n\n*************************Testing*************************")
    test_root = f"{args.datadir}/TestDataset/"
    test_metrics = test(model, test_root, args, binary_flag = True)
    
if __name__ == "__main__": 
    main()