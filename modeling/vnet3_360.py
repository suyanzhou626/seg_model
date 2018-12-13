import torch
import torch.nn as nn
from collections import OrderedDict
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
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
        input1 = torch.nn.functional.interpolate(input1,size=input2.size()[2:],mode='bilinear')
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
        self.args = args
        if self.args.sync_bn and 'rank' in self.args:
            import linklink as link
            def BNFunc(*args, **kwargs):
                return link.nn.SyncBatchNorm2d(*args, **kwargs, 
                                   group=self.args.bn_group, 
                                   sync_stats=True, 
                                   var_mode=self.args.bn_var_mode)
            BN = BNFunc
        elif self.args.sync_bn:
            BN = SynchronizedBatchNorm2d
        else:
            BN = nn.BatchNorm2d
        self.block_parameter = BLOCK_PARAMETER if block_parameter is None else block_parameter
        self.fcblock_parameter = FC_BLOCK if fcblock_parameter is None else fcblock_parameter 
        self.deconv_parameter = DECONV_PARAMETER
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
        for m in self.named_modules():
            if 'rank' in self.args:
                import linklink as link
                if isinstance(m[1],nn.BatchNorm2d) or isinstance(m[1],nn.PReLU) or isinstance(m[1],link.nn.SyncBatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
            else:
                if isinstance(m[1],nn.BatchNorm2d) or isinstance(m[1],nn.PReLU) or isinstance(m[1],SynchronizedBatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 3
    args.sync_bn = False
    model = Vnet3_360(args)
    model.eval()
    input = torch.rand(1, 3, 513,513)
    output = model(input)
    print(output.size())