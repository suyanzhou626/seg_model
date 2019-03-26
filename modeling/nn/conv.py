import torch
from torch.nn import Module

class Conv2d_Avg(Module):
    # this class is add avgpooling layer before dilation conv operation to capture partial information
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,
                groups=1,bias=True):
        super().__init__()
        self.pooling_size = dilation - 1 + dilation % 2
        self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,
                dilation,groups,bias)
    
    def forward(self,input):
        x = input
        # print(input.shape)
        if self.pooling_size > 1:
            # print('using pooling')
            x = torch.nn.functional.avg_pool2d(x,self.pooling_size,stride=1,
                        padding=self.pooling_size // 2)
            # print(x.shape)
        x = self.conv(x)
        return x

class Conv2d_Max(Module):
    # this class is add maxpooling layer before dilation conv operation to capture partial information
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,padding=0,dilation=1,
                groups=1,bias=True):
        super().__init__()
        self.pooling_size = dilation - 1 + dilation % 2
        self.conv = torch.nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,
                dilation,groups,bias)
    
    def forward(self,input):
        x = input
        # print(input.shape)
        if self.pooling_size > 1:
            # print('using pooling')
            x = torch.nn.functional.max_pool2d(x,self.pooling_size,stride=1,
                        padding=self.pooling_size // 2)
            # print(x.shape)
        x = self.conv(x)
        return x