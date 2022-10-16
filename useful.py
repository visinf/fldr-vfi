import numpy as np
import cupy as cp
import time 
import torch.nn.functional as F
class ScaleIt():

	def __init__(self, name,arr,free_axes):
		self.name = name
		self.arr = arr
		self.free_axes = free_axes	

		self.maxes = torch.zeros(self.arr.size()[:-2],dtype=torch.float64,device=arr.get_device())
		self.mins = torch.zeros(self.arr.size()[:-2],dtype=torch.float64,device=arr.get_device())
		for i in range(arr.size(0)):
			for k in range(arr.size(1)):
				if(self.free_axes == 3):
					for p in range(arr.size(2)):
						self.maxes[i,k,p] =torch.max(arr[i,k,p,:,:])
						self.mins[i,k,p] =torch.min(arr[i,k,p,:,:])
				else:
					self.maxes[i,k] =torch.max(arr[i,k,:,:])
					self.mins[i,k] =torch.min(arr[i,k,:,:])

		self.maxes = self.maxes.unsqueeze(self.free_axes).unsqueeze(self.free_axes+1)
		self.mins = self.mins.unsqueeze(self.free_axes).unsqueeze(self.free_axes+1)

	def scale(self, arr, n=1.0):
	    
	    return ((arr - self.mins)/(self.maxes-self.mins)).to(torch.float32)

	def backscale(self,arr):
		return ((arr *(self.maxes-self.mins))+ self.mins).to(torch.float32)

	def print_mins_maxes(self):
		print("Maxes: ",self.maxes.detach().cpu().numpy())
		print("Mins: ",self.mins.detach().cpu().numpy())

import torch

class MYPCA():

    def __init__(self,n_components=0):
        self.n_components = n_components
        self.store = dict()

    def fit_transform(self,data,device):
        
        self.fit(data.clone())
        transformed = self.transform(data,device)
        return transformed


    def fit(self,data):
        if(self.n_components==0):
            self.n_components = data.shape[1]
        self.n = data.shape[0]

        loc_data = data
        self.mean = torch.mean(data,dim=0)
        loc_data -=  self.mean

        some = True
        if(loc_data.shape[1] > 1023):
            some = False

        cupyuse = True
        if(cupyuse):
            with cp.cuda.Device(0):
                tempCup = cp.asarray(loc_data.clone().cpu())
                u,s,v = cp.linalg.svd(tempCup,full_matrices=False)
                u = 0
                s = torch.as_tensor(cp.asnumpy(s),device="cpu")
                v = torch.as_tensor(cp.asnumpy(v),device="cpu")
        else:
            u,s,v = torch.svd(loc_data,some=some)
        
        self.eigenvectors = v[:self.n_components,:]
        self.eigenvalues = (s**2)/self.n
        self.explained_variance_ratio_ =  self.eigenvalues/torch.sum(self.eigenvalues)
        


    def transform(self,data,device,compsused=0):
        
        self.mean = torch.as_tensor(self.mean,device=device)
        self.eigenvectors = torch.as_tensor(self.eigenvectors,device=device)
        loc_data = data - self.mean.to(device)
        if(compsused == 0):
            transformed = torch.matmul(loc_data,self.eigenvectors[:,:].permute(1,0).to(device))#.transpose(0,1) #cp.matmul(loc_data,cp.transpose(self.eigenvectors[:self.n_components,:]))
        else:
            transformed = torch.matmul(loc_data,self.eigenvectors[:compsused,:].permute(1,0).to(device))
        return transformed

    def inverse_transform(self,data):
        back = torch.matmul(data,self.eigenvectors[:,:])#.transpose(0,1)
        back = back + self.mean
        return back

    def store_sth(self,toSave,name):
        self.store[name] = toSave



from OpticalFlow.PWCNet import PWCNet
class MyPWC():
    def __init__(self,args,cpu=False):
        self.args = args
        self.flow_predictor = PWCNet()
        self.flow_predictor.load_state_dict(torch.load('./OpticalFlow/pwc-checkpoint.pt'))
        if(not cpu):
            self.flow_predictor.to(self.args.gpu)
    def get_flow(self,im0,im1):
        
        flow = self.flow_predictor(torch.cat([im0, im1], dim=0), torch.cat([im1, im0], dim=0))
        flow01, flow10 = torch.split(flow, im0.shape[0], dim=0)
        retflow = torch.cat([flow10,flow01],dim=1)
        return retflow

def distillation_loss(unref_flow_pyramid,gtflow,device):
    flowtop =  F.interpolate(unref_flow_pyramid[0], scale_factor=8,mode='bilinear',align_corners=False)
    flowshap = flowtop.shape
    pmap_10 = (-0.3 * torch.sqrt(torch.sum( (flowtop[:,:2,:].detach() - gtflow[:,:2,:])**2,dim=1,keepdim=True) )  ).exp()
    pmap_01 = (-0.3 * torch.sqrt(torch.sum((flowtop[:,2:,:].detach() - gtflow[:,2:,:])**2,dim=1,keepdim=True))  ).exp()

    alpha_10 = pmap_10 / 2
    alpha_01 = pmap_01 / 2
    epsilon_10 = 10 ** (-(10 * pmap_10 - 1) / 3)
    epsilon_01 = 10 ** (-(10 * pmap_01 - 1) / 3)
    gtflow_10 = gtflow[:,:2,:]
    gtflow_01 = gtflow[:,2:,:]
    retloss = torch.tensor(0.0,device=device)
    for temind,i in enumerate(unref_flow_pyramid[1:]):
        temflow_10 =  F.interpolate(i[:,:2,:], size=(flowshap[-2],flowshap[-2]),mode='bilinear',align_corners=False)
        temflow_01 =  F.interpolate(i[:,2:,:], size=(flowshap[-2],flowshap[-2]),mode='bilinear',align_corners=False)

        diff = temflow_10 - gtflow_10
        loss_10 = ((diff**2 + epsilon_10**2)**alpha_10).mean()
        retloss += loss_10

        diff = temflow_01 - gtflow_01
        loss_01 = ((diff**2 + epsilon_01**2)**alpha_01).mean()
        retloss += loss_01
    
    return retloss

def torch_prints(arr,name=""):
	print("------------------------ ARRAY " + name+" PRINTS ------------------------")
	print("shape: ",arr.shape)
	print("min: ", torch.min(arr))
	print("max: ", torch.max(arr))
	print("mean: ", torch.mean(arr))
	print("std: ", torch.std(arr))

	
def numpy_prints(arr,name=""):
	print("------------------------ ARRAY " + name+" PRINTS ------------------------")
	print("shape: ",arr.shape)
	print("min: ", np.min(arr))
	print("max: ", np.max(arr))
	print("mean: ", np.mean(arr))
	print("std: ", np.std(arr))

def getmodelconfig(args):
    args.pcanet = True
    args.mean_vector_norm = True
    args.ds_normInput = True
    args.scales = [8,16,32,64]
    args.fractions = [4,16,64,256]
    args.S_trn = 3
    args.S_tst = 3
    args.dataset = "X4K1000FPS"
    args.oneEV = True
    args.ref_feat_extrac = True
    args.optimizeEV = True
    args.lr_milestones = [70,120,170]
    args.ExacOneEV = True
    args.takeBestModel = True
    args.allImUp = True
    args.softsplat = True
    args.forwendflowloss = True
    args.warp_alpha = 0.05
    args.sminterp = True
    args.ownsmooth = True
    args.noResidAddup = True
    args.impmasksoftsplat = True
    args.cutoffUnnec = True
    args.fixsmoothtwistup = True
    args.sminterpInpIm = True
    args.patch_size = 512
    args.tempbottomflowfix = True
