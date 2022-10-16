import torch
import math
#from pac import PacConv2d,PacConvTranspose2d
from myAdditions import torch_prints
import torch.nn.functional as F
class MyConv2dPac(torch.nn.Module):
  def __init__(self,inC,outC,kernel=3,stride=1,padding=1):

    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn((outC,inC,kernel,kernel)))
    self.bias = torch.nn.Parameter(torch.randn((outC)))
    self.inC = inC
    self.outC = outC
    self.kernel = kernel
    self.padding = padding
    self.stride = stride
    self.reset_parameters()

  def forward(self, x,guide):
    B,C,H,W = x.shape
    _,g_ch,H2,W2 = guide.shape
    assert C == self.inC
    assert H2 == H
    assert W2 == W

    inp_unf = torch.nn.functional.unfold(x, kernel_size=self.kernel,padding=self.padding,stride=self.stride)
    guide_unf = torch.nn.functional.unfold(guide, kernel_size=self.kernel,padding=self.padding,stride=self.stride).reshape(B,g_ch,self.kernel**2,H//self.stride,W//self.stride)

    guide_unf = guide_unf - guide_unf[:,:,4,:,:].unsqueeze(2)
    #print(guide_unf.shape)
    guide_unf = self.kern_fun(guide_unf) # self.kern_fun(torch.zeros(guide_unf.shape))#self.kern_fun(guide_unf)
    #print(guide_unf.shape)
    #print(inp_unf.shape)
    inp_unf = (inp_unf.view(B,C,self.kernel**2,H//self.stride,W//self.stride) * guide_unf).view(B,C*self.kernel**2,H*W//self.stride**2)
    #print("after mul: ",inp_unf.shape)
    out_unf = (inp_unf.transpose(1, 2).matmul(self.weight.view(self.weight.size(0), -1).t()).reshape(B,H*W//self.stride**2,self.outC) + self.bias).transpose(1, 2).reshape(B,self.outC,H*W//self.stride**2)
    #print(out_unf.shape)
    out = out_unf.view(B, self.outC, H//self.stride, W//self.stride)
    return out#guide_unf.reshape(B,1,3,3,H,W) #out
  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
    fan_in,_ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1/math.sqrt(fan_in)
    torch.nn.init.uniform_(self.bias,-bound,bound)

  def kern_fun(self,x):
    return torch.exp(-0.5*torch.sum(x*x,dim=1)).unsqueeze(1)

class MyConv2dPacTranspose(torch.nn.Module):
  def __init__(self,inC,outC,kernel=3,stride=1,padding=1,output_padding=0):

    super().__init__()
    self.weight = torch.nn.Parameter(torch.randn((inC,outC,kernel,kernel)))
    self.bias = torch.nn.Parameter(torch.randn((outC)))
    self.inC = inC
    self.outC = outC
    self.kernel = kernel
    self.padding = padding
    self.stride = stride
    self.output_padding = output_padding
    self.reset_parameters()

  def forward(self, x,guide):
    B,C,H,W = x.shape
    _,g_ch,GH,GW = guide.shape
    assert C == self.inC
    assert GH == H * self.stride
    assert GW == W * self.stride
    guide_padding = (self.kernel - 1) * 1 // 2 
    guide_unf = torch.nn.functional.unfold(guide, kernel_size=self.kernel,padding=guide_padding,stride=1).reshape(B,g_ch,self.kernel**2,H*self.stride,W*self.stride)
    guide = 0
    #with torch.no_grad():
    #    self.weight = torch.nn.Parameter(torch.rot90(self.weight,2,[2,3]))
    guide_unf = guide_unf - guide_unf[:,:,self.kernel**2//2,:,:].unsqueeze(2)
    #print(guide_unf.shape)
    guide_unf = self.kern_fun(guide_unf).reshape(B,1,self.kernel,self.kernel,GH,GW) # self.kern_fun(torch.zeros(guide_unf.shape))#self.kern_fun(guide_unf)
    #return guide_unf.reshape(B,1,self.kernel,self.kernel,H*self.stride,W*self.stride)
    inp_unf = torch.nn.functional.unfold(x, kernel_size=self.kernel,padding=self.padding,stride=self.stride)

    w = x.new_ones((C, 1, 1, 1))
    temp = F.conv_transpose2d(x, w, stride=self.stride, groups=C)
    pad = (self.kernel - 1) * 1 - self.padding       
    temp = F.pad(temp, (pad, pad + self.output_padding, pad, pad + self.output_padding))

    cols = F.unfold(temp, self.kernel, 1, 0, 1)

    #print(guide_unf.shape)
    in_mul_k = cols.view(B, C, *guide_unf.shape[2:]) * guide_unf

    #print(in_mul_k.shape,self.weight.shape)
    # 8,2,3,3,4,6           2,8,3,3
    # matrix multiplication, written as an einsum to avoid repeated view() and permute()
    output = torch.einsum('ijklmn,jokl->iomn', (in_mul_k, self.weight))

    output += self.bias.view(1, -1, 1, 1)

    return output#.clone()  # TODO understand why a .clone() is needed here

  def reset_parameters(self):
    torch.nn.init.kaiming_uniform_(self.weight,a=math.sqrt(5))
    fan_in,_ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
    bound = 1/math.sqrt(fan_in)
    torch.nn.init.uniform_(self.bias,-bound,bound)

  def kern_fun(self,x):
    return torch.exp(-0.5*torch.sum(x*x,dim=1)).unsqueeze(1)


# def pacConv2dTesting():
#     B,C,H,W = (1,3,4,6)
#     stride = 2
#     inp = torch.randn(B, C, H, W)
#     guide = inp.clone()#torch.zeros(inp.shape)
#     #pacconv = PacConv2d(3,8,3,padding=1,stride=stride)
    
#     convit = MyConv2dPac(3,8,3,stride,1)# kernel,stride,padding
#     convit.w = pacconv.weight
#     convit.bias = pacconv.bias

#     out = convit(inp,guide)
#     #theirs = pacconv(inp, guide)
#     print((out - theirs ).abs().max()) 
#     #print(out)
#     #print(theirs)
#     torch_prints(out,"My")
#     torch_prints(theirs,"theirs")

#     #print((torch.nn.functional.conv2d(inp, convit.w,padding=1,bias=convit.bias) - out).abs().max()) 




#          ch = input.shape[1]
#         w = input.new_ones((ch, 1, 1, 1))
#         x = F.conv_transpose2d(input, w, stride=stride, groups=ch)
#         pad = [(kernel_size[i] - 1) * dilation[i] - padding[i] for i in range(2)]
#         x = F.pad(x, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
#         output = pacconv2d(x, kernel, weight.permute(1, 0, 2, 3), bias, dilation=dilation,
#                            shared_filters=shared_filters, native_impl=True)

if __name__ == '__main__' :
    B,C,H,W = (8,5,4,6)
    stride = 2
    inC = C
    outC = 8
    kernel = 3
    outputPad = 1
    padding = 1
    inp = torch.randn(B, C, H, W)
    guide = F.interpolate(inp.clone(),scale_factor=stride) #torch.zeros(inp.shape)#

    convit = MyConv2dPacTranspose(inC=inC,outC=outC,kernel=kernel,stride=stride,padding=padding,output_padding=outputPad)# kernel,stride,padding
    out = convit(inp,guide)

    pacconv = PacConvTranspose2d(inC,outC,kernel,padding=padding,stride=stride,output_padding=outputPad)
    pacconv.weight = convit.weight
    pacconv.bias = convit.bias
    theirs = pacconv(inp, guide)

    #convit.weight = pacconv.weight
    #convit.bias = pacconv.bias
    #pacconv.weight = convit.weight
    #pacconv.bias = convit.bias

    out = convit(inp,guide)
    
    print((out - theirs ).abs().max()) 
    #print(out)
    #print(theirs)
    torch_prints(out,"My")
    torch_prints(theirs,"theirs")

    #print((torch.nn.functional.conv2d(inp, convit.w,padding=1,bias=convit.bias) - out).abs().max()) 