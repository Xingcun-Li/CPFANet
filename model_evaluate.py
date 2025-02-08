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
from lib.PraNet_Res2Net import PraNet
from lib.ssformers import mit_PLD_b2
from lib.PFENet import PFENet
from lib.CGMA.model import CGMA
from lib.CPFANet import CPFANet
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from utils.metrics import calculate_metrics
from utils.dataloader import TestDataset
from utils.util import save_mask

def get_args():
    parser = argparse.ArgumentParser(description="Args for Polyp Segmentation")
    parser.add_argument("--dataset", type=str,
                        default="polyp5", help="DATASETNAME")
    parser.add_argument("--datadir", type=str,
                        default="./data/polypSegDataset", help="Dir of your dataset, like '/data/DATASETNAME'")
    parser.add_argument("--save_root", type=str,
                        default="./result_maps/CPFANet", help="Dir for saving test predictions")
    parser.add_argument("--num_workers", type=int,
                        default=16, help="Number of workers for loading the data")
    parser.add_argument("--batch_size", type=int,
                        default=16, help="Batch size, default=16")
    parser.add_argument("--augmentation", type=bool,
                        default=False, help="augmentation: use or not")
    parser.add_argument("--image_size", type=int,
                        default=352, help="Image size, default=352")
    parser.add_argument("--seed", type=int,
                        default=42, help="Random seed, default=42")
    parser.add_argument("--epochs", type=int,
                        default=40, help="Max training epochs, default=70")
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
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    return args

class DiceLoss(torch.nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1e-6
        intersection = (pred * target).sum(dim=(1, 2, 3))
        dice_score = (2. * intersection + smooth) / (pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + smooth)
        return 1 - dice_score.mean()

def evaluate(model, args, weight_path):
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    dataset_metrics = {data_name: {"mDice": 0.0, "mIoU": 0.0, "Precision": 0.0, "Recall": 0.0, "F2": 0.0,
                                   "S-measure": 0.0, "F-measure": 0.0, "E-measure": 0.0, "MAE": 0.0} 
                       for data_name in ['Kvasir', 'CVC-ClinicDB', 'CVC-ColonDB', 'ETIS-LaribPolypDB', 'CVC-300']}

    test_root = f"{args.datadir}/TestDataset/"
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
                preds = model(images)
                preds = preds
                preds_sigmoid = torch.sigmoid(preds)
                preds_binary = preds_sigmoid > 0.5

                calc_scores = calculate_metrics(preds_binary, gts)
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

        print(f"Dataset: {_data_name} - "
              f"mDice: {dataset_metrics[_data_name]['mDice']:.4f}, "
              f"mIoU: {dataset_metrics[_data_name]['mIoU']:.4f}, Precision: {dataset_metrics[_data_name]['Precision']:.4f}, "
              f"Recall: {dataset_metrics[_data_name]['Recall']:.4f}, F2: {dataset_metrics[_data_name]['F2']:.4f}, "
              f"S-measure: {dataset_metrics[_data_name]['S-measure']:.4f}, "
              f"F-measure: {dataset_metrics[_data_name]['F-measure']:.4f}, "
              f"E-measure: {dataset_metrics[_data_name]['E-measure']:.4f}, MAE: {dataset_metrics[_data_name]['MAE']:.4f}")

    return dataset_metrics

if __name__ == "__main__":
    args = get_args()
    model = CPFANet().cuda()
    weight_path = "./snapshots/CPFANet/model_epoch_40.pth" 
    metrics = evaluate(model, args, weight_path)