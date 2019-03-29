import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import sys
sys.path.append(os.getcwd())
from torch.nn import init
from collections import OrderedDict
from .nn.xception import Xception
from .nn.aspp import ASPP
from utils.load import load_pretrained_mode

model_urls = {
    'xception': '/mnt/lustre/wuyao/.torch/models/xception-b5690688.pth'#'http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth'
}
class DeepLabv3plus(nn.Module):
	def __init__(self, args):
		super(DeepLabv3plus, self).__init__()
		self.backbone = None		
		self.backbone_layers = None
		BatchNorm = args.batchnorm_function
		input_channel = 2048		
		self.aspp = ASPP(dim_in=input_channel, 
				dim_out=256, 
				rate=1,
				bn_mom = 0.0003,BatchNorm=BatchNorm)
		self.dropout1 = nn.Dropout(0.5)
		# self.upsample4 = nn.UpsamplingBilinear2d(scale_factor=4)
		# self.upsample_sub = nn.UpsamplingBilinear2d(scale_factor=4)

		indim = 256
		self.shortcut_conv = nn.Sequential(
				nn.Conv2d(indim, 48, 1, 1, padding=0,bias=True),
				BatchNorm(48, momentum=0.0003),
				nn.ReLU(inplace=True),		
		)		
		self.cat_conv = nn.Sequential(
				nn.Conv2d(304, 256, 3, 1, padding=1,bias=True),
				BatchNorm(256, momentum=0.0003),
				nn.ReLU(inplace=True),
				nn.Dropout(0.5),
				nn.Conv2d(256, 256, 3, 1, padding=1,bias=True),
				BatchNorm(256, momentum=0.0003),
				nn.ReLU(inplace=True),
				nn.Dropout(0.1),
		)
		self.cls_conv = nn.Conv2d(256, args.num_classes, 1, 1, padding=0)
		# for m in self.modules():
		# 	if isinstance(m, nn.Conv2d):
		# 		nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		self.backbone = Xception(os=16,BatchNorm=BatchNorm)
		self.backbone_layers = self.backbone.get_layers()
		for m in self.modules():
			classname = m.__class__.__name__
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()
			elif classname.find('BatchNorm') != -1:
				m.weight.data.fill_(1)
				if m.bias is not None:
					m.bias.data.zero_()
		if args.ft and args.resume is None:
			_ , _ , _ =load_pretrained_mode(self.backbone,checkpoint_path=model_urls['xception'])


	def forward(self, x):
		x_bottom = self.backbone(x)
		layers = self.backbone.get_layers()
		feature_aspp = self.aspp(layers[-1])
		feature_aspp = self.dropout1(feature_aspp)
		feature_aspp = F.interpolate(feature_aspp,size=layers[0].size()[2:],mode='bilinear',align_corners=True)
		# feature_aspp = self.upsample_sub(feature_aspp)

		feature_shallow = self.shortcut_conv(layers[0])
		feature_cat = torch.cat([feature_aspp,feature_shallow],1)
		result = self.cat_conv(feature_cat) 
		result = self.cls_conv(result)
		# result = self.upsample4(result)
		return result

	def get_conv_weight_params(self):
		for m in self.named_modules():
			if isinstance(m[1],nn.Conv2d):
				for p in m[1].named_parameters():
					if p[1].requires_grad:
						if p[0] == 'weight':
							yield p[1]

	def get_conv_bias_params(self):
		for m in self.named_modules():
			if isinstance(m[1],nn.Conv2d):
				for p in m[1].named_parameters():
					if p[1].requires_grad:
						if p[0] == 'bias':
							yield p[1]

	def get_bn_prelu_params(self):
		for m in self.named_parameters():
			if 'bn' in m[0] or 'relu' in m[0]:
				if m[1].requires_grad:
					yield m[1]

if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 2
    args.batchnorm_function = torch.nn.BatchNorm2d
    args.resume = None
    args.ft = True
    model = DeepLabv3plus(args)
    tempa = []
    tempb = []
    for i in model.named_parameters():
        tempa.append(i[0])

    for m in model.named_modules():
        if isinstance(m[1],torch.nn.Conv2d):
            for p in m[1].named_parameters():
                if p[1].requires_grad:
                    tempb.append(m[0]+'.'+p[0])
    for m in model.named_parameters():
        if 'bn' in m[0] or 'relu' in m[0]:
            if m[1].requires_grad:
                tempb.append(m[0])
    tempa = sorted(tempa)
    tempb = sorted(tempb)
    print(tempa == tempb)
    for i in tempa:
    	if i not in tempb:
    		print(i)
    num = 0
    for i in model.parameters():
        num += 1
    num1 = 0
    for i in model.get_conv_weight_params():
        num1 += 1
    for i in model.get_conv_bias_params():
        num1 += 1
    for i in model.get_bn_prelu_params():
        num1 += 1
    print(num,num1)
    model.eval()
    input = torch.rand(1, 3, 225,225)
    output = model(input)
    print(output.size())