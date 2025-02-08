import torch
import torch.nn as nn
import torch.nn.functional as F
from .pvtv2 import pvt_v2_b2
import math

# Hybrid Feature Enhancement
class HFE(nn.Module):
# Will be uploaded once the review is completed

# Polyp Contextual Perception Module
class PCP(nn.Module):
# Will be uploaded once the review is completed

# Multi-Scale Adaptive Aggregation Module
class MSAA(nn.Module):
# Will be uploaded once the review is completed

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.MSAA12  = MSAA(128)
        self.MSAA23  = MSAA(64)
        self.MSAA34  = MSAA(48)
        self.upconv1 = nn.Conv2d(256, 128, 3, 1, 1)
        self.upconv2 = nn.Conv2d(256, 64, 3, 1, 1)
        self.upconv3 = nn.Conv2d(128, 48, 3, 1, 1)
        self.choose1 = nn.Conv2d(256, 1, 1)
        self.choose2 = nn.Conv2d(128, 1, 1)
        self.choose3 = nn.Conv2d(64, 1, 1)
        self.choose4 = nn.Conv2d(48, 1, 1)
        self.choose = nn.Conv2d(96, 1, 1)

    def forward(self, x1, x2, x3, x4): 
        x1_init = x1
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x1 = self.upconv1(x1)
        x2 = self.MSAA12(x2,x1)
        x2 = torch.cat((x1, x2), dim=1)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x2 = self.upconv2(x2)
        x3 = self.MSAA23(x3,x2)
        x3 = torch.cat((x2, x3), dim=1)
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear')
        x3 = self.upconv3(x3)
        x4 = self.MSAA34(x4,x3)
        x4 = torch.cat((x3, x4), dim=1)
        return self.choose(x4), self.choose4(x3), self.choose3(x2), self.choose2(x1), self.choose1(x1_init)


class CPFANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pvt = pvt_v2_b2()
        path = './pretrained_weights/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.pvt.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.pvt.load_state_dict(model_dict)
        
        self.HFE1 = HFE(64,64)
        self.HFE2 = HFE(128,128)
        self.HFE3 = HFE(320,320)
        self.HFE4 = HFE(512,512)
        
        self.PCP1 = PCP(64,48)
        self.PCP2 = PCP(128,64)
        self.PCP3 = PCP(320,128)
        self.PCP4 = PCP(512,256)
        
        self.decoder = Decoder()

    def forward(self, x):
        x1, x2, x3, x4 = self.pvt(x)  # x1：B,64,H/4,H/4   x2：B,128,H/8,H/8   x3：B,320,H/16,H/16   x4：B,512,H/32,H/32
        x1 = self.HFE1(x1) # B,64,H/4,H/4
        x2 = self.HFE2(x2) # B,128,H/8,H/8
        x3 = self.HFE3(x3) # B,320,H/16,H/16
        x4 = self.HFE4(x4) # B,512,H/32,H/32
        x1 = self.PCP1(x1)  # B,48,H/4,H/4
        x2 = self.PCP2(x2)  # B,64,H/8,H/8
        x3 = self.PCP3(x3)  # B,128,H/16,H/16
        x4 = self.PCP4(x4)  # B,256,H/32,H/32
        decoder_map,decoder_map2,decoder_map3,decoder_map4,decoder_map5 = self.decoder(x4, x3, x2, x1) # B,1,H/4,W/4
        seg_map2 = F.interpolate(decoder_map2, scale_factor=4, mode='bilinear')
        seg_map3 = F.interpolate(decoder_map3, scale_factor=8, mode='bilinear')
        seg_map4 = F.interpolate(decoder_map4, scale_factor=16, mode='bilinear')
        seg_map5 = F.interpolate(decoder_map5, scale_factor=32, mode='bilinear')
        seg_map = F.interpolate(decoder_map, scale_factor=4, mode='bilinear') # B,1,H,W
        return seg_map, seg_map2, seg_map3, seg_map4, seg_map5

if __name__ == "__main__":
    input_tensor = torch.randn(16, 3, 352, 352)
    model = CPFANet()
    output = model(input_tensor)
    print(f"output:{[o.shape for o in output]}")