# ----------------------------------------
# Written by Yude Wang
# ----------------------------------------

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import gc
from torch.nn import init
from collections import OrderedDict
from .nn.xception import Xception
from .nn.aspp import ASPP

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
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
		self.backbone = Xception(os=16,BatchNorm=BatchNorm)
		self.backbone_layers = self.backbone.get_layers()


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
