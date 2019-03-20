import torch
import torch.nn as nn
from collections import OrderedDict
BN = None
BLOCK_PARAMETER=[{'layers':4,'planes':[3,32,32,32,32],'k_sizes':[7,3,3,3],'strides':[4,1,1,1],'pads':[3,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':4,'planes':[32,64,64,64,64],'k_sizes':[3,3,3,3],'strides':[2,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':4,'planes':[64,128,128,128,128],'k_sizes':[3,3,3,3],'strides':[2,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':4,'planes':[128,256,256,256,256],'k_sizes':[3,3,3,3],'strides':[2,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':4,'planes':[256,256,256,256,256],'k_sizes':[3,3,3,3],'strides':[1,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]}]
FC_BLOCK={'layers':3,'planes':[48,64,64,64],'k_sizes':[1,1,1],'strides':[1,1,1],'pads':[0,0,0],'dilations':[1,1,1]}

DECONV_PARAMETER=[{'layers':2,'planes':[512,128,128],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},
                  {'layers':2,'planes':[256,64,64],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},
                  {'layers':2,'planes':[128,32,32],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},
                  {'layers':2,'planes':[64,32,32],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]},]

def conv_bn_relu(inplane,outplane,k_size,stride=1,pad=0,dilation=1):
    block = nn.Sequential(OrderedDict([('conv',nn.Conv2d(inplane,outplane,k_size,stride=stride,padding=pad,dilation=dilation)),
    ('bn',BN(outplane)),('prelu',nn.PReLU(outplane))]))
    return block

class DeconvBlock(nn.Module):
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None,sync_bn=True):
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
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None,sync_bn=True):
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

class Vnet3_360(nn.Module):
    def __init__(self,args,block_parameter=None,fcblock_parameter=None,**kwards):
        super().__init__()
        global BN
        BN = args.batchnorm_function
        self.block_parameter = BLOCK_PARAMETER if block_parameter is None else block_parameter
        self.fcblock_parameter = FC_BLOCK if fcblock_parameter is None else fcblock_parameter 
        self.deconv_parameter = DECONV_PARAMETER
        if args.gray_mode:
            self.block_parameter[0]['planes'][0] = 1
        self.block1 = Block(**self.block_parameter[0])
        self.block2 = Block(**self.block_parameter[1])
        self.block3 = Block(**self.block_parameter[2])
        self.block4 = Block(**self.block_parameter[3])
        self.block5 = Block(**self.block_parameter[4])
        self.deconvbblock1 = DeconvBlock(**self.deconv_parameter[0])
        self.deconvbblock2 = DeconvBlock(**self.deconv_parameter[1])
        self.deconvbblock3 = DeconvBlock(**self.deconv_parameter[2])
        self.deconvbblock4 = DeconvBlock(**self.deconv_parameter[3])
        self.last_conv = nn.Conv2d(self.deconv_parameter[3]['planes'][-1],args.num_classes,1,stride=1,padding=0)
        self.branch1 = conv_bn_relu(self.block_parameter[0]['planes'][-1],self.block_parameter[0]['planes'][-1],1)
        self.branch2 = conv_bn_relu(self.block_parameter[1]['planes'][-1],self.block_parameter[1]['planes'][-1],1)
        self.branch3 = conv_bn_relu(self.block_parameter[2]['planes'][-1],self.block_parameter[2]['planes'][-1],1)
        self.branch4 = conv_bn_relu(self.block_parameter[3]['planes'][-1],self.block_parameter[3]['planes'][-1],1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
    def forward(self,input):
        x = self.block1(input)
        branch1 = self.branch1(x)
        x = self.block2(x)
        branch2 = self.branch2(x)
        x = self.block3(x)
        branch3 = self.branch3(x)
        x = self.block4(x)
        branch4 = self.branch4(x)
        # print(branch4.size())
        x = self.block5(x)
        x = self.deconvbblock1(x,branch4)
        x = self.deconvbblock2(x,branch3)
        x = self.deconvbblock3(x,branch2)
        x = self.deconvbblock4(x,branch1)
        x = self.last_conv(x)
        x = torch.nn.functional.interpolate(x,size=input.size()[2:],mode='bilinear',align_corners=True)
        return x

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
            if 'bn' in m[0] or 'prelu' in m[0]:
                if m[1].requires_grad:
                    yield m[1]

if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 2
    args.batchnorm_function = torch.nn.BatchNorm2d
    model = Vnet3_360(args)
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
        if 'bn' in m[0] or 'prelu' in m[0]:
            if m[1].requires_grad:
                tempb.append(m[0])
    tempa = sorted(tempa)
    tempb = sorted(tempb)
    print(tempa == tempb)
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
    input = torch.rand(1, 3, 481,481)
    output = model(input)
    print(output.size())