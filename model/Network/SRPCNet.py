#!/usr/bin/python3
#coding=utf-8

import torch
from .vgg_encoder import VGG
from .modules import SPA, HAIM, BSFM, ConvBR
import torch.nn as nn
import torch.nn.functional as F
from utils.tensor_ops import cus_sample, upsample_add


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Identity)):
            pass


class SRPCNet(torch.nn.Module):
    def __init__(self, cfg, model_name='SRPC'):
        super(SRPCNet, self).__init__()
        self.cfg = cfg
        self.model_name = model_name
        
        if self.model_name == "SRPC":
            #### VGG Encoder ####
            self.encoder = VGG()

        else:
            print("UNDEFINED BACKBONE NAME.")

        #SPA
        self.spa1 = SPA(64, 64)
        self.spa2 = SPA(128, 64)
        self.spa3 = SPA(256, 64)
        self.spa4 = SPA(512, 64)
        self.spa5 = SPA(512, 64)

        #HAIM
        self.haim1 = HAIM()
        self.hiam2 = HAIM()
        self.hiam3 = HAIM()
        self.haim4 = HAIM()
        self.hiam5 = HAIM()

        #BSFM
        self.upsample_add = upsample_add
        self.upsample = cus_sample
        self.bsfm1 = BSFM(64, 32)
        self.bsfm2 = BSFM(64, 32)
        self.bsfm3 = BSFM(64, 32)
        self.bsfm4 = BSFM(64, 32)
        self.bsfm5 = BSFM(64, 32)
        self.upconv1 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv2 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv3 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)
        self.upconv5 = ConvBR(64, 64, kernel_size=3, stride=1, padding=1)

        #Predtrans
        self.predtrans1 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans5 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, shape=None):

        features = self.encoder(x)

        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        # SPA
        x1 = self.spa1(x1)
        x2 = self.spa2(x2)
        x3 = self.spa3(x3)
        x4 = self.spa4(x4)
        x5 = self.spa5(x5)

        # HAIM
        x5 = self.hiam5(in1=x5, in2=x4)
        x4 = self.hiam4(in1=x4, in2=x5, in3=x3)
        x3 = self.hiam3(in1=x3, in2=x4, in3=x2)
        x2 = self.hiam2(in1=x2, in2=x3, in3=x1)
        x1 = self.hiam1(in1=x1, in2=x2)

        # BSFM
        x5 = self.bsfm5(x5)
        x5_up = self.upconv5(x5)

        x4 = self.upsample_add(x5_up, x4)
        x4 = self.bsfm4(x4)
        x4_up = self.upconv4(x4)

        x3 = self.upsample_add(x4_up, x3)
        x3 = self.bsfm3(x3)
        x3_up = self.upconv3(x3)

        x2 = self.upsample_add(x3_up, x2)
        x2 = self.bsfm2(x2)
        x2_up = self.upconv2(x2)

        x1 = self.upsample_add(x2_up, x1)
        x1 = self.bsfm1(x1)
        x1_up = self.upconv1(x1)

        if shape is None:
            shape = x.size()[2:]

        pred1 = F.interpolate(self.predtrans1(x1_up), size=shape, mode='bilinear')
        pred2 = F.interpolate(self.predtrans2(x2_up), size=shape, mode='bilinear')
        pred3 = F.interpolate(self.predtrans3(x3_up), size=shape, mode='bilinear')
        pred4 = F.interpolate(self.predtrans4(x4_up), size=shape, mode='bilinear')
        pred5 = F.interpolate(self.predtrans5(x5_up), size=shape, mode='bilinear')

        return pred1, pred2, pred3, pred4, pred5

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)

if __name__ == '__main__':
  
    import torch
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
       net = SRPCNet()
       macs, params = get_model_complexity_info(net, (3, 352, 352), as_strings=True, print_per_layer_stat=True, verbose=True)
       print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
       print('{:<30}  {:<8}'.format('Number of parameters: ', params))
