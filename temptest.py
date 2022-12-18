import torch

a = torch.randn(100)

print(torch.cuda.device_count())
print(torch.cuda.current_device())
a.to(0)
print(a)
