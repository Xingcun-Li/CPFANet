import torch
import numpy as np
from thop import profile
from thop import clever_format
from PIL import Image
import random

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay

def save_mask(pred, save_path):
    if isinstance(pred, np.ndarray):
        pred = pred.squeeze(0)
    else:
        pred = pred.squeeze(0).cpu().numpy()

    pred = (pred * 255).astype(np.uint8)
    pred_image = Image.fromarray(pred)
    pred_image.save(save_path)
    
def CalParams(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))

