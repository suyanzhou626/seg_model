import torch
import torch.nn as nn
from collections import OrderedDict
import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu1 = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.relu2 = nn.PReLU(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu3 = nn.PReLU(planes * 4)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, output_stride, BatchNorm):
        self.inplanes = 64
        super(ResNet, self).__init__()
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            blocks = [1, 2, 4]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            blocks = [1, 2, 1]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.PReLU(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        feature_4x = x
        x = self.layer2(x)
        feature_8x = x
        x = self.layer3(x)
        feature_16x = x
        x = self.layer4(x)
        return x, feature_16x,feature_8x,feature_4x

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

BN = None
# BLOCK_PARAMETER=[{'layers':2,'planes':[3,8,12],'k_sizes':[2,3],'strides':[2,2],'pads':[1,1],'dilations':[1,1]},
#                  {'layers':4,'planes':[12,12,12,12,12],'k_sizes':[3,3,3,3],'strides':[1,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
#                  {'layers':4,'planes':[12,24,24,24,24],'k_sizes':[3,3,3,3],'strides':[2,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
#                  {'layers':3,'planes':[24,32,32,32],'k_sizes':[3,3,3],'strides':[2,1,1],'pads':[1,1,1],'dilations':[1,1,1]},
#                  {'layers':4,'planes':[32,32,32,32,48],'k_sizes':[3,3,3,3],'strides':[1,1,1,1],'pads':[1,1,1,1],'dilations':[1,1,1,1]},
#                  {'layers':2,'planes':[18,8,12],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]}]
FC_BLOCK={'layers':4,'planes':[48,48,64,64,64],'k_sizes':[3,1,1,1],'strides':[1,1,1,1],'pads':[0,0,0,0],'dilations':[1,1,1,1]}

DECONV_PARAMETER=[{'layers':2,'planes':[64,32,32],'k_sizes':[3,3],'strides':[1,1],'pads':[1,1],'dilations':[1,1]}]

def conv_bn_relu(inplane,outplane,k_size,stride=1,pad=0,dilation=1):
    block = nn.Sequential(OrderedDict([('conv',nn.Conv2d(inplane,outplane,k_size,stride=stride,padding=pad,dilation=dilation)),
    ('bn',BN(outplane)),('prelu',nn.PReLU(outplane))]))
    return block

class DeconvPath(nn.Module):
    def __init__(self,layers,planes,k_sizes,strides=None,pads=None,dilations=None):
        super().__init__()
        self.strides = [1]*layers if strides==None else strides
        self.pads = [0]*layers if pads==None else pads
        self.dilations = [1]*layers if dilations==None else dilations
        blocks = []
        for i in range(layers):
            blocks.append(conv_bn_relu(planes[i],planes[i+1],k_sizes[i],stride=self.strides[i],pad=self.pads[i],dilation=self.dilations[i]))
        self.block = nn.ModuleList(blocks)
    def forward(self,x,ori_input):
        for i,module in enumerate(self.block):
            x = module(x)
            x = nn.functional.interpolate(x,scale_factor=2,mode='bilinear',align_corners=True)
        x = nn.functional.interpolate(x,size=ori_input.size()[2:],mode='bilinear',align_corners=True)
        x = nn.functional.softmax(x,dim=1)
        return x

class MSC(nn.Module):
    def __init__(self,args,**kwards):
        super().__init__()
        global BN
        BN = args.batchnorm_function
        self.deconv_parameter = DECONV_PARAMETER[0]
        self.deconv_parameter['planes'][-1]=args.num_classes
        self.encoder = ResNet(Bottleneck, [3, 4, 23, 3], 16, BN)
        self.deconv_parameter = DECONV_PARAMETER
        self.deconvblock = DeconvPath(**self.deconv_parameter[0])
        
        self.branch1 = conv_bn_relu(256,64,3,stride=1,pad=1,dilation=1)
        self.branch2 = conv_bn_relu(512,64,3,stride=1,pad=1,dilation=1)
        self.branch3 = conv_bn_relu(1024,64,3,stride=1,pad=1,dilation=1)
        self.branch4 = conv_bn_relu(2048,64,3,stride=1,pad=1,dilation=1)
        self.select_conv = conv_bn_relu(256,4,3,stride=1,pad=1,dilation=1)
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
        self.encoder._load_pretrained_model()
    def forward(self,input):
        x,feat_16,feat_8,feat_4 = self.encoder(input)
        x = self.branch4(x)
        feat_16 = self.branch3(feat_16)
        feat_8 = self.branch2(feat_8)
        feat_4 = self.branch1(feat_4)
        x_temp = nn.functional.interpolate(x,input.size()[2:],mode='bilinear',align_corners=True)
        feat16_temp = nn.functional.interpolate(feat_16,input.size()[2:],mode='bilinear',align_corners=True)
        feat8_temp = nn.functional.interpolate(feat_8,input.size()[2:],mode='bilinear',align_corners=True)
        feat4_temp = nn.functional.interpolate(feat_4,input.size()[2:],mode='bilinear',align_corners=True)
        select = self.select_conv(torch.cat([x_temp,feat16_temp,feat8_temp,feat4_temp],1))
        select = nn.functional.softmax(select,dim=1)
        select = torch.unsqueeze(select,dim=4)
        select = torch.transpose(select,1,4)
        score1 = self.deconvblock(x,input)
        score1 = torch.unsqueeze(score1,dim=4)
        score2 = self.deconvblock(feat_16,input)
        score2 = torch.unsqueeze(score2,dim=4)
        score3 = self.deconvblock(feat_8,input)
        score3 = torch.unsqueeze(score3,dim=4)
        score4 = self.deconvblock(feat_4,input)
        score4 = torch.unsqueeze(score4,dim=4)
        score = torch.cat([score1,score2,score3,score4],4)
        score = torch.mul(score,select)
        out = torch.sum(score,4)
        return out

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
            if m[1].__class__.__name__.find('BatchNorm2d') != -1 or isinstance(m[1],nn.PReLU):
                for p in m[1].named_parameters():
                    if p[1].requires_grad:
                        yield p[1]
    # def get_bn_prelu_params(self):
    #     for m in self.named_parameters():
    #         if 'bn' in m[0] or 'prelu' in m[0]:
    #             if m[1].requires_grad:
    #                 yield m[1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 4
    def BNFunc(*args, **kwargs):
        return nn.BatchNorm2d(*args,**kwargs)
    args.batchnorm_function = BNFunc
    model = MSC(args)
    tempa = []
    tempb = []
    for i in model.named_parameters():
        tempa.append(i[0])

    for m in model.named_modules():
        # print(m[0],' ',m[1].__class__.__name__)
        if isinstance(m[1],torch.nn.Conv2d):
            for p in m[1].named_parameters():
                if p[1].requires_grad:
                    tempb.append(m[0]+'.'+p[0])
    for m in model.named_modules():
        if m[1].__class__.__name__.find('BatchNorm2d') != -1 or isinstance(m[1],nn.PReLU):
            for p in m[1].named_parameters():
                if p[1].requires_grad:
                    tempb.append(m[0]+'.'+p[0])
    tempa = sorted(tempa)
    tempb = sorted(tempb)
    print(len(tempa),len(tempb))
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
    input = torch.rand(1, 3, 513,513)
    output1= model(input)
    print(output1.size())