import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

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
                               dilation=dilation, padding=dilation, bias=False)
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
                nn.SeparableConv2d(self.inplanes, planes * block.expansion,
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

def conv_bn_relu(inplane,outplane,k_size,stride=1,dilation=1):
    block = nn.Sequential(OrderedDict([('conv',SeparableConv2d(inplane,outplane,k_size,stride=stride,dilation=dilation)),
    ('bn',BN(outplane)),('prelu',nn.PReLU(outplane))]))
    return block

class MIML(nn.Module):
    def __init__(self,args,**kwards):
        super().__init__()
        global BN
        BN = args.batchnorm_function
        self.conv_in = nn.Conv2d(3,32,3,stride=2,padding=1)
        self.bn_in = BN(32)
        self.prelu_in = nn.PReLU(32)
        self.boundary_layer1 = conv_bn_relu(32,64,3)
        self.boundary_layer2 = conv_bn_relu(64,48,3,stride=2)
        self.boundary_layer3 = conv_bn_relu(48,2,3)

        self.semantic_layer1 = conv_bn_relu(48,64,3,stride=2)
        self.semantic_layer2 = conv_bn_relu(64,64,3)
        self.semantic_layer3 = conv_bn_relu(64,64,3)
        self.semantic_layer4 = conv_bn_relu(64,96,3,stride=2)
        self.semantic_layer5 = conv_bn_relu(96,96,3)
        self.semantic_layer6 = conv_bn_relu(96,96,3)
        self.semantic_layer7 = conv_bn_relu(96,128,3,stride=2)
        self.semantic_layer8 = conv_bn_relu(128,128,3)
        self.semantic_layer9 = conv_bn_relu(128,128,3)
        self.adaptiveavgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128,256)
        self.fc2 = nn.Linear(256,128)
        self.fc_class = nn.Linear(128,args.num_classes)
        self.fusion_layer1 = conv_bn_relu(128,48,3)
        self.output1 = conv_bn_relu(96,48,3)
        self.output2 = conv_bn_relu(48,args.num_classes,3)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d')!= -1:
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self,input):
        x = self.conv_in(input)
        x = self.bn_in(x)
        x = self.prelu_in(x)
        x = self.boundary_layer1(x)
        x = self.boundary_layer2(x)
        boundary_out = self.boundary_layer3(x)
        
        y = x.detach()
        y = self.semantic_layer1(y)
        y = self.semantic_layer2(y)
        y = self.semantic_layer3(y)
        y = self.semantic_layer4(y)
        y = self.semantic_layer5(y)
        y = self.semantic_layer6(y)
        y = self.semantic_layer7(y)
        y = self.semantic_layer8(y)
        y = self.semantic_layer9(9)
        z = self.adaptiveavgpool(y)
        z = torch.squeeze(z)
        z = self.fc1(z)
        z = self.fc2(z)
        y = 
        z = self.fc_class(z)
        y = self.fusion_layer1(y)
        y = self.output1(torch.cat([x,y],dim=1))
        y = self.output2(y)

        return y,boundary_out,z
