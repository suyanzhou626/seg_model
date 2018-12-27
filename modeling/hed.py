import torch
import torchvision
from torch import nn
# input size [256,256]
# 基于vgg16 hed
BN = None

def make_layers(blocks):
    layers = []
    in_channels = 3
    for v in blocks:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, BN(v), nn.PReLU(v)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class HED_vgg16(nn.Module):
    def __init__(self,args,num_filters=32, pretrained=False):
        # Here is the function part, with no braces ()
        super().__init__()
        global BN
        BN = args.batchnorm_function
        encoder = make_layers(cfg['D'])
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv1=encoder[0:6]
        self.score1=nn.Sequential(nn.Conv2d(num_filters*2,1,1,1),nn.PReLU(1))# 256*256
        
        self.conv2=encoder[7:13]
        self.d_conv2=nn.Sequential(nn.Conv2d(num_filters*4,1,1,1),nn.PReLU(1))#128*128
        
        self.conv3=encoder[14:23]
        self.d_conv3=nn.Sequential(nn.Conv2d(num_filters*8,1,1,1),nn.PReLU(1))#64*64
        
        self.conv4=encoder[24:33]
        self.d_conv4=nn.Sequential(nn.Conv2d(num_filters*16,1,1,1),nn.PReLU(1))#32*32
        
        self.conv5=encoder[34:43]
        self.d_conv5=nn.Sequential(nn.Conv2d(num_filters*16,1,1,1),nn.PReLU(1))#16*16
        
        self.score=nn.Conv2d(5,args.num_classes,1,1)# No relu
        
    def forward(self,x):
        # Here is the part that calculates the return value
        x=self.conv1(x)
        s1=self.score1(x)
        x=self.pool(x)

        x=self.conv2(x)
        s_x=self.d_conv2(x)
        s2=torch.nn.functional.interpolate(s_x,size=s1.size()[2:],mode='bilinear',align_corners=True)
        x=self.pool(x)

        x=self.conv3(x)
        s_x=self.d_conv3(x)
        s3=torch.nn.functional.interpolate(s_x,size=s1.size()[2:],mode='bilinear',align_corners=True)
        x=self.pool(x)
        
        x=self.conv4(x)
        s_x=self.d_conv4(x)
        s4=torch.nn.functional.interpolate(s_x,size=s1.size()[2:],mode='bilinear',align_corners=True)
        x=self.pool(x)
        
        x=self.conv5(x)
        s_x=self.d_conv5(x)
        s5=torch.nn.functional.interpolate(s_x,size=s1.size()[2:],mode='bilinear',align_corners=True)
        
        score=self.score(torch.cat([s1,s2,s3,s4,s5],dim=1))
        
        return score

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

if __name__ == "__main__":
    import argparse
    import torch
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_classes = 2
    args.batchnorm_function = torch.nn.BatchNorm2d
    model = HED_vgg16(args)
    tempa = []
    tempb = []
    for i in model.named_parameters():
        tempa.append(i[0])

    for m in model.named_modules():
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
    output = model(input)
    print(output.size())