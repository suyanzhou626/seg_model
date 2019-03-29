import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

BN = None

def conv_bn_relu(inplane,outplane,k_size,stride=1,pad=0,dilation=1):
    block = nn.Sequential(OrderedDict([('conv',nn.Conv2d(inplane,outplane,k_size,stride=stride,padding=pad,dilation=dilation)),
    ('bn',BN(outplane)),('prelu',nn.PReLU(outplane))]))
    return block

class Block(nn.Module):
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None):
        super(Block,self).__init__()
        self.strides = [[1],[1,1],[1,1],[1,1]] if strides==None else strides
        self.pads = [0]*layers if pads==None else pads
        self.dilations = [[1],[1,1],[1,1],[1,1]] if dilations==None else dilations
        block = []
        for j in range(4):
            blocks = []
            for i in range(layers[j]):
                blocks.append(conv_bn_relu(planes[j][i],planes[j][i+1],k_sizes[j][i],stride=self.strides[j][i],pad=self.pads[j][i],dilation=self.dilations[j][i]))
            block.append(nn.Sequential(*blocks))
        self.block = nn.ModuleList(block)

    def forward(self,input):
        output = []
        for i in range(4):
            outtemp = self.block[i](input)
            output.append(outtemp)
        output = torch.cat(output,dim=1)
        return output

class EncoderLayer(nn.Module):
    def __init__(self,main,branch):
        super(EncoderLayer,self).__init__()
        block_main = []
        block_branch = []
        self.pooling = nn.MaxPool2d(3,2,1)
        for i in range(len(main)):
            block_main.append(Block(**main[i]))
        for j in range(len(branch)):
            block_branch.append(Block(**branch[j]))
        self.block_branch = nn.Sequential(*block_branch)
        self.block_main = nn.Sequential(*block_main)
    
    def forward(self,input):
        output = input
        branch = input
        output = self.pooling(output)
        output = self.block_main(output)
        branch = self.block_branch(branch)
        return output,branch

class DecoderLayer(nn.Module):
    def __init__(self,main):
        super(DecoderLayer,self).__init__()
        block_main = []
        for i in range(len(main)):
            block_main.append(Block(**main[i]))
        self.block_main = nn.Sequential(*block_main)
    
    def forward(self,input,low_feature):
        output = F.interpolate(input,size=low_feature.data.size()[2:],mode='bilinear',align_corners=True)
        output = torch.add(output,1,low_feature)
        output = self.block_main(output)
        return output

ENCODER1_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[128,32],[128,32,32],[128,32,32],[128,32,32]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[128,32],[128,32,32],[128,32,32],[128,32,32]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]}
],
'branch':[
    {'layers':[1,2,2,2],'planes':[[128,16],[128,64,16],[128,64,16],[128,64,16]],'k_sizes':[[1],[1,3],[1,7],[1,11]],'pads':[[0],[0,1],[0,3],[0,5]]}
]
}

ENCODER2_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[128,32],[128,32,32],[128,32,32],[128,32,32]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[128,64],[128,32,64],[128,32,64],[128,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]}
],
'branch':[
    {'layers':[1,2,2,2],'planes':[[128,32],[128,32,32],[128,32,32],[128,32,32]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[128,32],[128,64,32],[128,64,32],[128,64,32]],'k_sizes':[[1],[1,3],[1,7],[1,11]],'pads':[[0],[0,1],[0,3],[0,5]]}
]
}

ENCODER3_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]}
],
'branch':[
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,64],[256,64,64],[256,64,64],[256,64,64]],'k_sizes':[[1],[1,3],[1,7],[1,11]],'pads':[[0],[0,1],[0,3],[0,5]]}
]
}

ENCODER4_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]}
],
'branch':[
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]}
]
}

DECODER1_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,64],[256,64,64],[256,64,64],[256,64,64]],'k_sizes':[[1],[1,3],[1,7],[1,11]],'pads':[[0],[0,1],[0,3],[0,5]]}
]
}

DECODER2_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[256,64],[256,32,64],[256,32,64],[256,32,64]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[256,32],[256,32,32],[256,32,32],[256,32,32]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]}
]
}

DECODER3_PARAMETER = {
'main':[
    {'layers':[1,2,2,2],'planes':[[128,32],[128,32,32],[128,32,32],[128,32,32]],'k_sizes':[[1],[1,3],[1,5],[1,7]],'pads':[[0],[0,1],[0,2],[0,3]]},
    {'layers':[1,2,2,2],'planes':[[128,16],[128,64,16],[128,64,16],[128,64,16]],'k_sizes':[[1],[1,3],[1,7],[1,11]],'pads':[[0],[0,1],[0,3],[0,5]]}
]
}

class Hourglass(nn.Module):
    def __init__(self,args,**kwards):
        super(Hourglass,self).__init__()
        global BN
        BN = args.batchnorm_function
        self.entry = conv_bn_relu(3,128,3,pad=1)
        self.encoderlayer1 = EncoderLayer(**ENCODER1_PARAMETER)
        self.encoderlayer2 = EncoderLayer(**ENCODER2_PARAMETER)
        self.encoderlayer3 = EncoderLayer(**ENCODER3_PARAMETER)
        self.encoderlayer4 = EncoderLayer(**ENCODER4_PARAMETER)
        self.decoderlayer1 = DecoderLayer(**DECODER1_PARAMETER)
        self.decoderlayer2 = DecoderLayer(**DECODER2_PARAMETER)
        self.decoderlayer3 = DecoderLayer(**DECODER3_PARAMETER)
        self.exit = conv_bn_relu(64,args.num_classes,3,pad=1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,input):
        x = self.entry(input)
        x,branch1 = self.encoderlayer1(x)
        x,branch2 = self.encoderlayer2(x)
        x,branch3 = self.encoderlayer3(x)
        x,branch4 = self.encoderlayer4(x)
        x = self.decoderlayer1(x,branch4)
        x = self.decoderlayer2(x,branch3)
        x = self.decoderlayer3(x,branch2)
        x = F.interpolate(x,size=branch1.data.size()[2:],mode='bilinear',align_corners=True)
        x = torch.add(x,1,branch1)
        x = self.exit(x)
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
    model = Hourglass(args)
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
    input = torch.rand(1, 3, 225,225)
    output = model(input)
    print(output.size())