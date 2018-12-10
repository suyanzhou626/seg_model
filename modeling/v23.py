import torch
import torch.nn as nn
from collections import OrderedDict
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

BLOCK_PARAMETER=[{'layers':3,'planes':[3,8,12,12],'k_sizes':[2,3,3],'strides':[2,2,1],'pads':[1,1,1],'dilations':[1,1,1]},
                 {'layers':3,'planes':[12,12,12,12],'k_sizes':[3,3,3],'strides':[1,1,1],'pads':[1,1,1],'dilations':[1,1,1]},
                 {'layers':4,'planes':[12,24,24,24,24],'k_sizes':[3,3,3,3],'strides':[2,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
                 {'layers':3,'planes':[24,32,32,32],'k_sizes':[3,3,3],'strides':[2,1,1],'pads':[1,1,1],'dilations':[1,1,1]},
                 {'layers':4,'planes':[32,32,32,32,48],'k_sizes':[3,3,3,3],'strides':[1,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]}]
FC_BLOCK={'layers':3,'planes':[48,64,64,64],'k_sizes':[1,1,1],'strides':[1,1,1],'pads':[0,0,0],'dilations':[1,1,1]}

def conv_bn_relu(inplane,outplane,k_size,stride=1,pad=0,dilation=1,sync_bn=True):
    block = nn.Sequential(OrderedDict([('conv',nn.Conv2d(inplane,outplane,k_size,stride=stride,padding=pad,dilation=dilation)),
    ('bn',nn.BatchNorm2d(outplane) if not sync_bn else SynchronizedBatchNorm2d(outplane)),('prelu',nn.PReLU(outplane))]))
    return block

class Block(nn.Module):
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None,sync_bn=True):
        super(Block,self).__init__()
        self.strides = [1]*layers if strides==None else strides
        self.pads = [0]*layers if pads==None else pads
        self.dilations = [1]*layers if dilations==None else dilations
        blocks = []
        for i in range(layers):
            blocks.append(conv_bn_relu(planes[i],planes[i+1],k_sizes[i],stride=self.strides[i],pad=self.pads[i],dilation=self.dilations[i],sync_bn=sync_bn))
        self.block = nn.Sequential(*blocks)
    def forward(self,input):
        input = self.block(input)
        return input

class V23_4x(nn.Module):
    def __init__(self,nclasses,sync_bn=True,block_parameter=None,fcblock_parameter=None,**kwards):
        super(V23_4x,self).__init__()
        self.block_parameter = BLOCK_PARAMETER if block_parameter is None else block_parameter
        self.fcblock_parameter = FC_BLOCK if fcblock_parameter is None else fcblock_parameter 
        self.block1 = Block(**self.block_parameter[0],sync_bn=sync_bn)
        self.block2 = Block(**self.block_parameter[1],sync_bn=sync_bn)
        self.block3 = Block(**self.block_parameter[2],sync_bn=sync_bn)
        self.block4 = Block(**self.block_parameter[3],sync_bn=sync_bn)
        self.block5 = Block(**self.block_parameter[4],sync_bn=sync_bn)
        self.dilation1 = conv_bn_relu(self.block_parameter[4]['planes'][-1],self.block_parameter[4]['planes'][-1],3,pad=4,dilation=4,sync_bn=sync_bn)
        self.fc_block = Block(**self.fcblock_parameter,sync_bn=sync_bn)
        self.last_conv1 = nn.Conv2d(self.fcblock_parameter['planes'][-1],nclasses,1)
        self.branch = conv_bn_relu(self.block_parameter[1]['planes'][-1],nclasses,1,sync_bn=sync_bn)
        self.last_conv2 = nn.Conv2d(2*nclasses,nclasses,3,padding=1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self,input):
        x = self.block1(input)
        out = self.block2(x)
        x = self.block3(out)
        x = self.block4(x)
        x = self.block5(x)
        x = self.dilation1(x)
        x = self.fc_block(x)
        x = self.last_conv1(x)
        x = torch.nn.functional.interpolate(x,scale_factor=4,mode='bilinear',align_corners=True)
        out = self.branch(out)
        out = torch.nn.functional.interpolate(out,size=x.size()[2:],mode='bilinear',align_corners=True)
        x = torch.cat([x,out],dim=1)
        x = self.last_conv2(x)
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
        for m in self.named_modules():
            if isinstance(m[1],nn.BatchNorm2d) or isinstance(m[1],nn.PReLU) or isinstance(m[1],SynchronizedBatchNorm2d):
                for p in m[1].parameters():
                    if p.requires_grad:
                        yield p


if __name__ == "__main__":
    model = V23_4x(5)
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
    input = torch.rand(1, 3, 513,513)
    output = model(input)
    print(output.size())