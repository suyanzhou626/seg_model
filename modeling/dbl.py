import torch
import torch.nn as nn
from collections import OrderedDict

BLOCK_PARAMETER=[{'layers':2,'planes':[3,8,12],'k_sizes':[2,3],'strides':[2,2],'pads':[1,1],'dilations':[1,1]},
                 {'layers':4,'planes':[12,12,12,12,12],'k_sizes':[3,3,3,3],'strides':[1,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':4,'planes':[12,24,24,24,24],'k_sizes':[3,3,3,3],'strides':[2,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':3,'planes':[24,32,32,32],'k_sizes':[3,3,3],'strides':[2,1,1],'pads':[1,1,1],'dilations':[1,1,1]},
                 {'layers':4,'planes':[32,32,32,32,48],'k_sizes':[3,3,3,3],'strides':[1,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':3,'planes':[18,8,12,2],'k_sizes':[3,3,3],'strides':[1,1,1],'pads':[1,1,1],'dilations':[1,1,1]}]
FC_BLOCK={'layers':5,'planes':[48,48,64,64,64,2],'k_sizes':[3,1,1,1,1],'strides':[1,1,1,1,1],'pads':[0,0,0,0,0],'dilations':[1,1,1,1,1]}

# DECONV_PARAMETER=[{'layers':2,'planes':[512,128,128],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},
#                   {'layers':2,'planes':[256,64,64],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},
#                   {'layers':2,'planes':[128,32,32],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},
#                   {'layers':2,'planes':[64,32,32],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},]

def conv_bn_relu(inplane,outplane,k_size,stride=1,pad=0,dilation=1):
    block = nn.Sequential(OrderedDict([('conv',nn.Conv2d(inplane,outplane,k_size,stride=stride,padding=pad,dilation=dilation)),
    ('bn',nn.BatchNorm2d(outplane)),('prelu',nn.PReLU())]))
    return block

class DeconvBlock(nn.Module):
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None):
        super().__init__()
        self.strides = [1]*layers if strides==None else strides
        self.pads = [0]*layers if pads==None else pads
        self.dilations = [1]*layers if dilations==None else dilations
        blocks = []
        for i in range(layers):
            blocks.append(conv_bn_relu(planes[i],planes[i+1],k_sizes[i],stride=self.strides[i],pad=self.pads[i],dilation=self.dilations[i]))
        self.block = nn.Sequential(*blocks)
    def forward(self,input1,input2):
        input1 = torch.nn.functional.interpolate(input1,size=input2.size()[2:],mode='bilinear',align_corners=True)
        input = torch.cat([input1,input2],dim=1)
        input = self.block(input)
        return input

class Block(nn.Module):
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None):
        super().__init__()
        self.strides = [1]*layers if strides==None else strides
        self.pads = [0]*layers if pads==None else pads
        self.dilations = [1]*layers if dilations==None else dilations
        blocks = []
        for i in range(layers):
            blocks.append(conv_bn_relu(planes[i],planes[i+1],k_sizes[i],stride=self.strides[i],pad=self.pads[i],dilation=self.dilations[i]))
        self.block = nn.Sequential(*blocks)
    def forward(self,input):
        input = self.block(input)
        return input

class Dbl(nn.Module):
    def __init__(self,nclasses,block_parameter=None,fcblock_parameter=None,**kwards):
        super().__init__()
        self.block_parameter = BLOCK_PARAMETER if block_parameter is None else block_parameter
        self.fcblock_parameter = FC_BLOCK if fcblock_parameter is None else fcblock_parameter 
        self.block_parameter[-1]['planes'][-1]=nclasses
        self.block_parameter[-1]['planes'][0]=3*nclasses+self.block_parameter[0]['planes'][-1]
        self.fcblock_parameter['planes'][-1]=nclasses
        # self.deconv_parameter = DECONV_PARAMETER
        self.block1 = Block(**self.block_parameter[0])
        self.block2 = Block(**self.block_parameter[1])
        self.block3 = Block(**self.block_parameter[2])
        self.block4 = Block(**self.block_parameter[3])
        self.block5 = Block(**self.block_parameter[4])
        # self.deconvbblock1 = DeconvBlock(**self.deconv_parameter[0])
        # self.deconvbblock2 = DeconvBlock(**self.deconv_parameter[1])
        # self.deconvbblock3 = DeconvBlock(**self.deconv_parameter[2])
        # self.deconvbblock4 = DeconvBlock(**self.deconv_parameter[3])
        self.fc_block = Block(**self.fcblock_parameter)
        self.last_block = Block(**self.block_parameter[5])
        self.branch = conv_bn_relu(self.block_parameter[1]['planes'][-1],nclasses,1)
        self.fc_conv = nn.Conv2d(2*nclasses,nclasses,3,stride=1,padding=1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self,input):
        block1_1 = self.block1(input)
        x = self.block2(block1_1)
        branch = self.branch(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.fc_block(x)
        x = torch.nn.functional.interpolate(x,size=branch.size()[2:],mode='bilinear',align_corners=True)
        x = torch.cat([x,branch],dim=1)
        fc_conv_1 = self.fc_conv(x)
        out1 = torch.nn.functional.interpolate(fc_conv_1,size=input.size()[2:],mode='bilinear',align_corners=True)
        x = torch.cat([fc_conv_1,x,block1_1],dim=1)
        x = self.last_block(x)
        out2 = torch.nn.functional.interpolate(x,size=input.size()[2:],mode='bilinear',align_corners=True)
        return out1,out2

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


    def get_bn_params(self):
        for m in self.named_modules():
            if isinstance(m[1],nn.BatchNorm2d):
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p


if __name__ == "__main__":
    model = Dbl(5)
    model.eval()
    input = torch.rand(1, 3, 513,513)
    output1,output2 = model(input)
    print(output1.size(),output2.size())