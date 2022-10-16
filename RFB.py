import torch
import torch.nn as nn


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1, visual = 1,part_rfb=False):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        self.part_rfb = part_rfb
        inter_planes = in_planes // 8
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, 2*inter_planes, kernel_size=1, stride=stride),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual, dilation=visual, relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=visual+1, dilation=visual+1, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, (inter_planes//2)*3, kernel_size=3, stride=1, padding=1),
                BasicConv((inter_planes//2)*3, 2*inter_planes, kernel_size=3, stride=stride, padding=1),
                BasicConv(2*inter_planes, 2*inter_planes, kernel_size=3, stride=1, padding=2*visual+1, dilation=2*visual+1, relu=False)
                )

        self.ConvLinear = BasicConv((2 if(self.part_rfb)else 6)*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        torch.cuda.empty_cache()
        #print("OTHER PRINTS!!!!!!!")
        #print(x0.size(),x1.size(),x2.size())
        if(self.part_rfb):
            x0 = self.ConvLinear(x0)
            x1 = self.ConvLinear(x1)
            x2 = self.ConvLinear(x2)

            short = self.shortcut(x)

            outx0 = self.relu(x0*self.scale + short)
            outx1 = self.relu(x1*self.scale + short)
            outx2 = self.relu(x2*self.scale + short)

            return outx0,outx1,outx2
        else:
            out = torch.cat((x0,x1,x2),1)
            x0 = 0
            x1 = 0
            x2 = 0
            out = self.ConvLinear(out)
            short = self.shortcut(x)
            torch.cuda.empty_cache()
            
            out = out*self.scale + short
            out = self.relu(out)

            return out

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        # self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=[1,kernel_size,kernel_size], stride=[1,stride,stride], padding=[0,padding,padding], dilation=[1,dilation,dilation], groups=groups, bias=bias)
        # self.bn = nn.BatchNorm3d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=[kernel_size,kernel_size], stride=[stride,stride], padding=[padding,padding], dilation=[dilation,dilation], groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
