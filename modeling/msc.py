import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.utils.model_zoo as model_zoo

def fixed_padding(inputs, kernel_size, dilation):
    kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
    return padded_inputs

class SeparableConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size, stride, 0, dilation,
                               groups=inplanes, bias=bias)
        self.pointwise = nn.Conv2d(inplanes, planes, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = fixed_padding(x, self.conv1.kernel_size[0], dilation=self.conv1.dilation[0])
        x = self.conv1(x)
        x = self.pointwise(x)
        return x
BN = None

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = SeparableConv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.relu1 = nn.PReLU(planes)
        self.conv2 = SeparableConv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.relu2 = nn.PReLU(planes)
        self.conv3 = SeparableConv2d(planes, planes * 4, kernel_size=1, bias=False)
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
                SeparableConv2d(self.inplanes, planes * block.expansion,
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
        # feature_4x = x
        x = self.layer2(x)
        # feature_8x = x
        x = self.layer3(x)
        # feature_8x = x
        x = self.layer4(x)
        return x

    def _load_pretrained_model(self):
        pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

def conv_bn_relu(inplane,outplane,k_size,stride=1,dilation=1):
    block = nn.Sequential(OrderedDict([('conv',SeparableConv2d(inplane,outplane,k_size,stride=stride,dilation=dilation)),
    ('bn',BN(outplane)),('prelu',nn.PReLU(outplane))]))
    return block

class MSC(nn.Module):
    def __init__(self,args,**kwards):
        super().__init__()
        global BN
        BN = args.batchnorm_function
        self.encoder = ResNet(Bottleneck, [3, 4, 23, 3], 16, BN)
        self.foregroundlayer1 = conv_bn_relu(2048,128,3)
        self.foregroundlayer2 = conv_bn_relu(128,2,3)

        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.classifierlayer1 = nn.Linear(2048,512)
        self.classifierlayer2 = nn.Linear(512,args.num_classes)

        self.semanticlayer1 = conv_bn_relu(2048,512,3)
        self.semanticlayer2 = conv_bn_relu(512,64,3)
        self.semanticlayer3 = conv_bn_relu(64,args.num_classes,1)
        # self.deconvblock = DeconvPath(**self.deconv_parameter[0])
        
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1 and classname != 'SeparableConv2d':
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('Linear')!= -1:
                nn.init.kaiming_normal_(m.weight)
                m.bias.data.zero_()
        self.encoder._load_pretrained_model()
    def forward(self,input):
        x = self.encoder(input)
        y = x.detach()
        x = self.foregroundlayer1(x)
        x = self.foregroundlayer2(x)
        out_two = torch.argmax(x,dim=1,keepdim=True)
        out_two = out_two.type_as(y)
        temp = y * out_two
        z = temp.detach()
        y = self.adaptiveavgpool(y)
        y = y.view(y.size(0),-1)
        y = self.classifierlayer1(y)
        y = self.classifierlayer2(y)
        temp_y = y.detach()
        temp_y = torch.unsqueeze(temp_y,2)
        temp_y = torch.unsqueeze(temp_y,3)
        temp_y = torch.sigmoid(temp_y)
        z = F.interpolate(z,scale_factor=2,mode='bilinear',align_corners=True)
        z = self.semanticlayer1(z)
        z = F.interpolate(z,scale_factor=2,mode='bilinear',align_corners=True)
        z = self.semanticlayer2(z)
        z = F.interpolate(z,scale_factor=2,mode='bilinear',align_corners=True)
        z = self.semanticlayer3(z)
        z *= temp_y
        z = F.interpolate(z,size=input.data.size()[2:],mode='bilinear',align_corners=True)

        return [z,y,x]

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
    input = torch.rand(1, 3, 64,64)
    output1= model(input)
    print(output1[0].size())
    print(output1[1].size())
    print(output1[2].size())