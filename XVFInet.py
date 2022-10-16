import functools, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from RFB import BasicRFB
import math
from skimage import feature
import cv2
import sys
# My Imports
from myAdditions import torch_prints,numpy_prints,MyPWC
#from pca_comp import to_dctpca,from_dctpca_to_dct_diff,pca_inverse
from pca_comp import pca_inverse,to_pca_diff
#from pac import PacConvTranspose2d
from pacTesting import MyConv2dPac,MyConv2dPacTranspose
import cupy as cp
import time
from softSplat import Softsplat


class DCTXVFInet(nn.Module):
	
	def __init__(self, args):
		super(DCTXVFInet, self).__init__()
		self.args = args
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
		#self.nf = args.nf
		#self.scale = args.module_scale_factor
		self.lrelu = nn.ReLU()
		self.in_channels = args.img_ch
		
		self.output_size = (args.patch_size,args.patch_size)	
		#self.output_ds = (8,20,40)[:args.S_trn+1]	

		self.output_size_val = (self.args.validation_patch_size,self.args.validation_patch_size)
		self.output_size_test = (2160,4096)
		self.nf = int(self.args.dctvfi_nf * self.args.doubleParameters )

		self.base_modules = nn.ModuleList([])
		#	1x1	2x2  4x4, 	8x8
		#   4x4, 8x8, 16x16, 32x32
		if(self.args.ref_feat_extrac):
			self.rec_ctx_ds = nn.Sequential(
				nn.Conv2d(self.args.dctvfi_nf*6 + (2 if(self.args.cannyInput)else 0), self.nf*self.args.img_ch*2, 3, 1, 1),
				nn.ReLU(),
				nn.Conv2d(self.nf*6, self.args.dctvfi_nf*self.args.img_ch*2, 3, 1, 1),
				nn.ReLU()
				)
			
			if(self.args.extra4Layer):
				self.extra4Layer = BasicRFB(self.args.dctvfi_nf*self.args.img_ch*2,self.args.dctvfi_nf*self.args.img_ch*2,stride=1)
				self.base_modules.append(self.extra4Layer)
			self.base_modules.append(self.rec_ctx_ds)

		if(self.args.noPCA):
			self.prefeatextrac = nn.Sequential(
				nn.Conv2d(self.args.dctvfi_nf*6,self.args.dctvfi_nf*6 , 3, 1, 1),
				nn.ReLU(),
				nn.Conv2d(self.args.dctvfi_nf*6,self.args.dctvfi_nf*6 , 3, 1, 1),
				nn.ReLU()
				)
			self.downone = nn.Sequential(
				nn.Conv2d(self.args.dctvfi_nf*6 ,self.args.dctvfi_nf*6 , 3, 2, 1),
				nn.ReLU()
				)
			self.increaseMaps = nn.Sequential(
				nn.Conv2d(6,self.args.dctvfi_nf*6 , 3, 1, 1),
				nn.ReLU()
				)

		self.vfinet = DCTVFInet(args,self.output_size,self.output_size_test,self.output_size_val)
		self.base_modules.append(self.vfinet)
		if(self.args.addPWCOneFlow):
			self.mypwc = MyPWC(self.args)
			for param in self.mypwc.flow_predictor.parameters():
				param.requires_grad = False
		else:
			self.mypwc = None

		res = sum(p.numel() for p in self.vfinet.parameters())
		print("Parameters of DCTVFInet: ",res)
		
		if(self.args.optimizeEV):
			#self.pca_means = [torch.nn.Parameter(torch.zeros((self.args.scales[i]**2),device=self.args.gpu),requires_grad=True).double() for i in range(len(self.args.scales))]
			#self.EVs = [torch.nn.Parameter(torch.zeros(((self.args.scales[0]**2)//self.args.fractions[0],self.args.scales[i]**2),device=self.args.gpu),requires_grad=True).double() for i in range(len(self.args.scales))]
			local_scales = self.args.scales
			if(self.args.allImUp):
				local_scales = [8 for i in range(len(self.args.scales))]
			self.EV8 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf,local_scales[0]**2),device=self.args.gpu).double())
			self.EV16 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf,local_scales[1]**2),device=self.args.gpu).double())
			self.EV32 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf,local_scales[2]**2),device=self.args.gpu).double()) if(self.args.S_trn>1)else None
			self.EV64 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf,local_scales[3]**2),device=self.args.gpu).double()) if(self.args.S_trn>2)else None
			self.Mean8 = torch.nn.Parameter(torch.empty((local_scales[0]**2),device=self.args.gpu).double())
			self.Mean16 = torch.nn.Parameter(torch.empty((local_scales[1]**2),device=self.args.gpu).double())
			self.Mean32 = torch.nn.Parameter(torch.empty((local_scales[2]**2),device=self.args.gpu).double())	if(self.args.S_trn>1)else None
			self.Mean64 = torch.nn.Parameter(torch.empty((local_scales[3]**2),device=self.args.gpu).double())	if(self.args.S_trn>2)else None
			
			self.EV8.requires_grad = not self.args.noEVOptimization
			self.Mean8.requires_grad = not self.args.noEVOptimization
			if(self.args.imageUpInp):
				self.EV8 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf,8**2),device=self.args.gpu).double())
				self.Mean8 = torch.nn.Parameter(torch.empty((8**2),device=self.args.gpu).double())

			self.pca_means = [self.Mean8,self.Mean16,self.Mean32,self.Mean64]
			self.EVs = [self.EV8,self.EV16,self.EV32,self.EV64]
			self.mean_vecs = [0 for i in range(len(self.args.scales))]
			
			if(self.args.meanVecParam):
				self.meanVec8 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf),device=self.args.gpu).double())
				self.meanVec16 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf),device=self.args.gpu).double())
				self.meanVec32 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf),device=self.args.gpu).double()) if(self.args.S_trn>1)else None
				self.meanVec64 = torch.nn.Parameter(torch.empty((self.args.dctvfi_nf),device=self.args.gpu).double()) if(self.args.S_trn>2)else None
				self.mean_vecs = [self.meanVec8,self.meanVec16,self.meanVec32,self.meanVec64]
				for i in self.mean_vecs:
					if(i == None):
						continue
					i.requires_grad = False

			temp = [self.EV8,self.EV16,self.EV32,self.EV64,self.Mean8,self.Mean16,self.Mean32,self.Mean64]	
			self.ev_params = [i for i in temp if(i!= None)]

			# self.ev_params = nn.ModuleDict({
			# 	"EVs" : nn.ModuleList([self.EV8,self.EV16,self.EV32,self.EV64,self.Mean8,self.Mean16,self.Mean32,self.Mean64])
			# 	})

		self.used_pcas = None
		self.params = None
		print("The lowest scale depth for training (S_trn): ", self.args.S_trn)
		print("The lowest scale depth for test (S_tst): ", self.args.S_tst)
		#self.double()
	# def ev_params(self):
	# 	return chain(m.parameters() for m in [self.EV8,self.EV16,self.EV32,self.EV64,self.Mean8,self.Mean16,self.Mean32,self.Mean64])
	# def base_params(self):
	# 	return chain(m.parameters() for m in  [self.EV8,self.EV16,self.EV32,self.EV64,self.Mean8,self.Mean16,self.Mean32,self.Mean64])


# input_gpuList,all_pcas,params, t_value,normInput=input_gpu
	def forward(self, input_gpuList, t_value,normInput=0,is_training=True,validation=False,epoch=0,frameT=None):
		'''
		input_big shape : [B,C*2*wiS*wiS,H,W]
		input_small shape: [B,C*2*wiS_*wiS_ /8,H_,W_] channels = 300
		pcas[0],params[0]: small
		pcas[1],params[1]: big
		t_value shape : [B,1] ###############
		'''
		B, C,  H, W = input_gpuList[0].size()
		B,_,_,H2,W2 = normInput[0].size()
		B2, C2 = t_value.size()
		assert C2 == 1, "t_value shape is [B,]"
		#assert T % 2 == 0, "T must be an even number"
		t_value = t_value.view(B, 1, 1, 1)
		x_l = normInput
		orig_images = x_l[0]

		if(self.args.dsstart > 1):
			#print(int(np.log2(4)))
			for temin in range(int(np.log2(self.args.dsstart))):
				x_l.pop(0)
				temshapfrom = (B,2,3,x_l[-1].shape[-2],x_l[-1].shape[-1])
				x_l.append(F.interpolate(x_l[-1].permute(0,2,1,3,4).reshape(B*2,3,temshapfrom[-2],temshapfrom[-1]), scale_factor=1/ 2,mode='bicubic', 
	                        align_corners=self.args.align_cornerse).reshape(B,2,3,temshapfrom[-2]//2,temshapfrom[-1]//2).permute(0,2,1,3,4))
			B,_,_,H2,W2 = x_l[0].size()

		flow_l = None
		needed = max(self.args.S_tst,self.args.S_trn) + 1
		for i in range(len(input_gpuList) - needed):
			input_gpuList[i+needed] = None
		torch.cuda.empty_cache()

		#print("Before pca trans")
		#time.sleep(3)
		#for i in self.mean_vecs:
		#	print("mean vec: ",i)
		# PCA Conversion
		start_time = time.time()
		if(self.args.optimizeEV):
			for i in range(self.args.S_tst+1):
				input_gpuList[i] = torch.zeros(input_gpuList[i].shape,device=self.args.gpu).float()
				if(self.args.noPCA):
					input_gpuList[i] = self.increaseMaps(x_l[0].reshape(B,-1,H2,W2))
					if(i== 1):
						input_gpuList[i] = self.prefeatextrac(self.downone(input_gpuList[i]))
					else:
						input_gpuList[i] = self.prefeatextrac(input_gpuList[i])
				elif(i == 0 and self.args.imageUpInp):
					#print("input gpu lsit: ",input_gpuList[i].shape)
					tempInp =  F.interpolate(x_l[0].reshape(B,6,H2,W2),scale_factor=2,mode="nearest").reshape(B*6,H2*2,W2*2)
					#print(tempInp.shape,self.params[i])
					#print(to_pca_diff(tempInp,self.params[i],self.args,self.pca_means[i],self.EVs[i],self.mean_vecs[i]).shape,B,self.args.dctvfi_nf*6,H2//4,W2//4)
					input_gpuList[i] = to_pca_diff(tempInp,self.params[i],self.args,self.pca_means[i],self.EVs[i],self.mean_vecs[i]).reshape(B,self.args.dctvfi_nf*6,H2//4,W2//4).float()
				elif(self.args.ExacOneEV):
					tempMul = 8/self.args.scales[i]
					index8 = self.args.scales.index(8)
					#_,_,_,TH,TW = normInput[i].shape
					if(tempMul > 1):
						mode = "nearest"
					else:
						mode = "bilinear"

					# if(tempMul == 1):
					# 	tempInp =x_l[0].reshape(B,6,H2,W2).reshape(B*6,int(H2*tempMul),int(W2*tempMul))
					# else:
					# 	tempInp =  F.interpolate(x_l[0].reshape(B,6,H2,W2),scale_factor=tempMul,mode=mode).reshape(B*6,int(H2*tempMul),int(W2*tempMul))
					#print("TEMPINP: ",tempInp.shape,tempMul)
					#tempMul /= 8
					for temIndex,temEvs in enumerate(self.EVs):
						if(temEvs == None):
							continue
						assert not temEvs.isnan().any(), "Any EV "+str(temIndex)
					#print("shapes: ",x_l[i].reshape(B*6,int(H2*tempMul),int(W2*tempMul)).shape)
					# if(self.args.noEVOptimization):
					# 	self.pca_means[index8].requires_grad = False
					# 	self.EVs[index8].requires_grad = False
					if(self.args.noEVOptimization):
						assert not self.pca_means[index8].requires_grad and not self.EVs[index8].requires_grad, "requires grad is true!!!"
					input_gpuList[i] = to_pca_diff(x_l[i].reshape(B*6,int(H2*tempMul),int(W2*tempMul)),self.params[i],self.args,self.pca_means[index8],self.EVs[index8],self.mean_vecs[index8]).reshape(B,self.args.dctvfi_nf*6,int(H2*tempMul//8),int(W2*tempMul//8)).float()
				elif(self.args.allImUp and self.args.scales[i] != 8):
					tempMul = 8/self.args.scales[i]
					#_,_,_,TH,TW = normInput[i].shape
					if(tempMul > 1):
						mode = "nearest"
					else:
						mode = "bilinear"
					#tempInp =  F.interpolate(x_l[0].reshape(B,6,H2,W2),scale_factor=tempMul,mode=mode).reshape(B*6,int(H2*tempMul),int(W2*tempMul))
					tempMul /= 8
					#assert not tempInp.isnan().any(), "tempinp is nan"
					input_gpuList[i] = to_pca_diff(x_l[i],self.params[i],self.args,self.pca_means[i],self.EVs[i],self.mean_vecs[i]).reshape(B,self.args.dctvfi_nf*6,int(H2*tempMul),int(W2*tempMul)).float()
					assert not input_gpuList[i].isnan().any(), "input_gpuList is nan, i: "+str(i)
				else:
					input_gpuList[i] = to_pca_diff(x_l[0].reshape(B*6,H2,W2),self.params[i],self.args,self.pca_means[i],self.EVs[i],self.mean_vecs[i]).reshape(input_gpuList[i].shape).float()
					if(self.args.noEVOptimization):
						assert not self.pca_means[i].requires_grad and not self.EVs[i].requires_grad, "requires grad is true!!!"
					else:
						assert self.pca_means[i].requires_grad and self.EVs[i].requires_grad, "requires grad is false!!!"

		tempInp = 0
		torch.cuda.empty_cache()
		# print("after pca trans")
		# time.sleep(3)
		if(self.args.timetest):
			print("Time diff pca conv: ",time.time()-start_time)
		#print("EVS: ",self.EVs[1][0,:])
		#print("mean vector: ",self.pca_means[1])
		#print("Meanvec:",self.mean_vecs[1])

		if(self.args.meanVecParam):
			for i in self.mean_vecs:
				if(i == None):
					continue
				assert not i.requires_grad 

		start_time = time.time()
		# Feature Extraction
		feat_x_list = [] ############# GO ON HEERE
		for i in range(needed):
			#torch_prints(input_gpuList[i],"level 0")
			if(self.args.ref_feat_extrac):
				if(self.args.cannyInput):
					im0 = F.interpolate(x_l[i][:,:,0,:,:],scale_factor=1/self.args.scales[0],mode="bilinear").unsqueeze(2)
					im1 = F.interpolate(x_l[i][:,:,1,:,:],scale_factor=1/self.args.scales[0],mode="bilinear").unsqueeze(2)
					#print(im0.shape)
					extrtemp = torch.cat([im0,im1],dim=2)
					#print(extrtemp.shape)
					canny = 0.299* extrtemp[:,0] + 0.587 * extrtemp[:,1] + 0.114 * extrtemp[:,2]
					canny = (canny +1)/2
					#print(canny.shape)
					#torch_prints(canny)

					for bdim in range(B):
						for im in range(2):
	#									print(canny[bdim,im,:,:].unsqueeze(2).shape)
	#									cv2.imwrite(f'ownTextFiles/Orig{bdim}{im}.png', canny[bdim,im,:,:].cpu().numpy()*255)
							canny[bdim,im,:,:] = torch.as_tensor(feature.canny(canny[bdim,im,:,:].cpu().numpy(),sigma=3)).to(self.args.gpu)
	#									cv2.imwrite(f'ownTextFiles/{bdim}{im}.png', canny[bdim,im,:,:].cpu().numpy()*255)
	#									torch_prints(canny[bdim,im,:,:])

					feat_x_list.append(self.rec_ctx_ds(torch.cat([input_gpuList[i],canny],dim=1))+input_gpuList[i])
					if(self.args.extra4Layer and i==0):
						feat_x_list[-1] = self.extra4Layer(feat_x_list[-1])+feat_x_list[-1]
				else:
					feat_x_list.append(self.rec_ctx_ds(input_gpuList[i])+input_gpuList[i])
					#print("feat_x: ",i,feat_x_list[-1].shape)
					#time.sleep(3)
		if(self.args.timetest):
			print("Time for featextrac: ",time.time()-start_time)
		
		input_gpuList = 0 #[0 for i in range(self.args.S_trn+1)]
		torch.cuda.empty_cache()
		pwcflow_list = []
		if(self.args.addPWCOneFlow):
			imtem0 =  F.interpolate(x_l[0][:,:,0,:], scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
			imtem1 =  F.interpolate(x_l[0][:,:,1,:], scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
			with torch.no_grad():
				flow_pwc = self.mypwc.get_flow(imtem0,imtem1)
				flow_pwc = 0.5 * F.interpolate(flow_pwc, scale_factor=1/2, mode='bilinear', align_corners=self.args.align_cornerse)
			
			for i in range(needed):
				pwcflow_list.append(flow_pwc.clone())
				flow_pwc = 0.5 * F.interpolate(flow_pwc, scale_factor=1/2, mode='bilinear', align_corners=self.args.align_cornerse)
			#pwcflow_list = pwcflow_list[::-1]
		else:
			for i in range(needed):
				pwcflow_list.append(0)
		if is_training:
			out_l_list = []
			flow_refine_l_list = []
			unrefined_flow_list = []
			endflow_list = []

			if(self.args.S_trn > 0):
				out_l, flow_l, maskList,flow_refine_l, endflow = self.vfinet(feat_x_list[self.args.S_trn], flow_l, t_value, level=self.args.S_trn, is_training=True,normInput=x_l[self.args.S_trn],validation=validation,epoch=epoch,feat_pyr=pwcflow_list[self.args.S_trn],mypwc=self.mypwc,orig_images=orig_images)
				out_l_list.append(out_l)
				flow_refine_l_list.append(flow_refine_l)
				unrefined_flow_list.append(flow_l)
				endflow_list.append(endflow)
			
			for level in range(self.args.S_trn-1, 0, -1): ## self.args.S_trn, self.args.S_trn-1, ..., 1. level 0 is not included
				out_l, flow_l, maskList, flow_refine_l, endflow = self.vfinet(feat_x_list[level], flow_l, t_value, level=level, is_training=True,normInput=x_l[level],validation=validation,epoch=epoch,feat_pyr=pwcflow_list[level],orig_images=orig_images)
				out_l_list.append(out_l)
				flow_refine_l_list.append(flow_refine_l)
				unrefined_flow_list.append(flow_l)
				endflow_list.append(endflow)
				torch.cuda.empty_cache()

			#print("----------------- Second VFINET ------------------------")
			#print(len(feat_x_list),len(all_pcas),len(params),len(normInput))
			out_l, flow_l, flow_refine_l, occ_0_l0, endflow, refine_out = self.vfinet(feat_x_list[0], flow_l, t_value, level=0, is_training=True,normInput=x_l[0],validation=validation,epoch=epoch,feat_pyr=pwcflow_list[0],mypwc=None,orig_images=orig_images,frameT=frameT)
			out_l_list.append(out_l)
			flow_refine_l_list.append(flow_refine_l)
			unrefined_flow_list.append(flow_l)
			endflow_list.append(endflow)
			############### 	Close UP 		####################################################

			flow_refine_l_list = flow_refine_l_list[::-1]
			out_l_list = out_l_list[::-1]
			unrefined_flow_list =unrefined_flow_list[::-1]
			endflow_list = endflow_list[::-1]

			# Just that it works
			if(self.args.no_refine):
				occ_0_l0 = torch.zeros((B,1,self.args.patch_size,self.args.patch_size))
			if(self.args.pcanet):
				mean_pics = torch.mean(orig_images,dim=2)
			else:
				mean_pics = torch.zeros((B,3,self.args.patch_size,self.args.patch_size))
			return out_l_list, flow_refine_l_list,unrefined_flow_list, occ_0_l0, mean_pics,endflow_list,refine_out #torch.zeros((#torch.mean(x, dim=2) # out_l_list should be reversed. [out_l0, out_l1, ...]

		else: # Testing
			for level in range(self.args.S_tst, 0, -1): ## self.args.S_tst, self.args.S_tst-1, ..., 1. level 0 is not included
				start_time = time.time()
				flow_l = self.vfinet(feat_x_list[level], flow_l, t_value, level=level, is_training=False,normInput=x_l[level],validation=validation,feat_pyr=pwcflow_list[level],mypwc=self.mypwc,orig_images=orig_images)
				feat_x_list[level] = 0
				torch.cuda.empty_cache()
				if(self.args.timetest):
					print("Time for level ",level," :",time.time()-start_time)
			start_time = time.time()
			out_l,refined_flow = self.vfinet( feat_x_list[0], flow_l, t_value, level=0, is_training=False,normInput=x_l[0],validation=validation,feat_pyr=pwcflow_list[0],mypwc=[],orig_images=orig_images)
			if(self.args.timetest):
				print("Last level: ",time.time()-start_time)
			# Reverse Padding
			out_l = out_l[:,:,:self.output_size_test[0],:self.output_size_test[1]]
			return out_l, refined_flow

	def pick_pca(self,pca):
		if(self.used_pcas == None):
			self.used_pcas = pca
			if(self.args.simpleEVs):
				self.used_pcas.store["mima"] = (self.used_pcas.store["mima"][0].to(self.args.gpu),self.used_pcas.store["mima"][1].to(self.args.gpu))
				with torch.no_grad():
					if(self.args.phase=="train"):
						temp = torch.as_tensor(cp.asnumpy(self.used_pcas.mean.copy()),device=self.args.gpu)
						temp.requires_grad= not self.args.noEVOptimization
						self.pca_means[0][:] = temp
						assert not self.pca_means[0].isnan().any()
						self.pca_means[0].requires_grad = not self.args.noEVOptimization

						temp = torch.as_tensor(cp.asnumpy(self.used_pcas.eigenvectors.copy()),device=self.args.gpu)
						temp.requires_grad = not self.args.noEVOptimization

						self.EVs[0][:] = temp
						self.EVs[0].requires_grad = not self.args.noEVOptimization
						assert self.EVs[0].get_device()==self.args.gpu
						assert self.pca_means[0].get_device()==self.args.gpu

						assert not self.EVs[0].isnan().any()
					self.mean_vecs[0][:] = torch.as_tensor(self.used_pcas.store["mean_vec"].copy(),device=self.args.gpu)
					assert self.mean_vecs[0].get_device()==self.args.gpu
			else:
				for i in self.used_pcas:
					i.store["mima"] = (i.store["mima"][0].to(self.args.gpu),i.store["mima"][1].to(self.args.gpu))
				if(self.args.optimizeEV): #self.alpha = t.nn.Parameter(t.tensor(0.5), requires_grad=True).cuda()
					index = 0
					
					with torch.no_grad():
						for i in self.used_pcas:
							if(self.args.phase=="train"):
								temp = torch.as_tensor(cp.asnumpy(i.mean),device=self.args.gpu)
								temp.requires_grad= not self.args.noEVOptimization
								#print(temp.shape,self.pca_means[index].shape)
								if(not self.args.ExacOneEV or index==0): # MAYBE CORRECT IT TODO
									self.pca_means[index][:] = temp 
									self.pca_means[index].requires_grad = not self.args.noEVOptimization
								temp = torch.as_tensor(cp.asnumpy(i.eigenvectors),device=self.args.gpu)
								temp.requires_grad= not self.args.noEVOptimization
								if(not self.args.ExacOneEV or index==0):
									#print(index,i.mean.shape)
									self.EVs[index][:] = temp
									self.EVs[index].requires_grad = not self.args.noEVOptimization
								
							if(not self.args.meanVecParam):
								self.mean_vecs[index] = torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)
							else:
								if(not self.args.ExacOneEV or index==0):
									self.mean_vecs[index][:] = torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)

							#self.diff_pca.append({"EV": torch.nn.Parameter(torch.as_tensor(i.eigenvectors,device=self.args.gpu),requires_grad=True), "Mean": torch.nn.Parameter(torch.as_tensor(i.mean,device=self.args.gpu),requires_grad=True), "MeanVec": torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)})
							#self.diff_pca.append({"EV": torch.as_tensor(i.eigenvectors,device=self.args.gpu), "Mean": torch.as_tensor(i.mean,device=self.args.gpu), "MeanVec": torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)})
							#3_068_948
							#self.pca_means[index].requires_grad=True
							#self.EVs[index].requires_grad=True
							index += 1
		self.used_pcas = None
	def pick_norm_vec(self,pca):
		if(self.used_pcas == None):
			self.used_pcas = pca
			for i in self.used_pcas:
				i.store["mima"] = (i.store["mima"][0].to(self.args.gpu),i.store["mima"][1].to(self.args.gpu))
			if(self.args.optimizeEV): #self.alpha = t.nn.Parameter(t.tensor(0.5), requires_grad=True).cuda()
				index = 0
				with torch.no_grad():
					for i in self.used_pcas:
						if(not self.args.meanVecParam):
							self.mean_vecs[index] = torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)
						else:
							self.mean_vecs[index][:] = torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)
						index += 1
		self.used_pcas = None

	def save_params(self,params):
		if(self.params == None):
			self.params = params
			for i in params:
				i.weightMat = i.weightMat.to(self.args.gpu)


class DCTVFInet(nn.Module):
	
	def __init__(self, args,output_size,output_size_test,output_size_val):
		super(DCTVFInet, self).__init__()
		self.args = args
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
		self.nf = int(self.args.dctvfi_nf * self.args.img_ch * self.args.doubleParameters) # 96/2
		self.scale = args.module_scale_factor
		self.in_channels = 3
		self.output_size = output_size
		self.output_size_test = output_size_test
		self.output_size_val = output_size_val		
		# 2*pad -(4-1) - 1 + 1
		# Beginning BFlowNet


		if(self.args.softsplat):
			self.softsplat = Softsplat()
		
		if self.args.pwcresid or self.args.additivePWC:
			self.mypwc = MyPWC(self.args)
			for param in self.mypwc.flow_predictor.parameters():
				param.requires_grad = False
		if(self.args.pwcbottomflow or self.args.justpwcflow ):
			self.mypwc = MyPWC(self.args)
			for param in self.mypwc.flow_predictor.parameters():
				param.requires_grad = False
		else:
			self.conv_flow_bottom = nn.Sequential( 
				nn.Conv2d(2*self.args.dctvfi_nf * self.args.img_ch + (4 if(self.args.additivePWC or self.args.addPWCOneFlow)else 0), 2*self.nf, [3,3], 1, [1,1]), # kernelsize,stride,padding
				nn.ReLU(),
				nn.Conv2d(2*self.nf, 2*self.nf, [3,3], 1, [1,1]), 
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf,4 if(self.args.cutoffUnnec and not self.args.tempbottomflowfix)else 6, [3,3], 1, [1,1]), #4 if(self.args.cutoffUnnec)else 6
				)

		# BFlowNet
		self.conv_flow1 = nn.Conv2d(2 *self.args.dctvfi_nf * self.args.img_ch, self.nf, [3, 3], 1, [1, 1])

		if(self.args.cutoffUnnec):
			self.conv_flow2 = nn.Sequential(
				nn.Conv2d(2*self.nf + 4, 2 * self.nf, [3, 3], 1, [1,1]), 
				nn.ReLU(),
				nn.Conv2d(2 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(1 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
			)
		else:
			self.conv_flow2 = nn.Sequential(
				nn.Conv2d(2*self.nf + 4, 2 * self.nf, [3, 3], 1, [1,1]), 
				nn.ReLU(),
				nn.Conv2d(2 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(1 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 6, [3, 3], 1, [1, 1]),
			)

		if(not self.args.cutoffUnnec or self.args.pwcflowrefine ):
			inputmaps = 4 + self.args.dctvfi_nf * 3 * 4
			if(self.args.ignoreFeatx):
				inputmaps = 4
			if(self.args.complallrefine):
				inputmaps = 200
			if(self.args.pwcflowrefine):
				inputmaps += 4
			if(self.args.pwcresid):
				inputmaps += 16
			self.conv_flow3 = nn.Sequential(
				nn.Conv2d(inputmaps, self.nf, [1, 1], 1, [0, 0]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(2 * self.nf, 4 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
			)
		if(self.args.pwcresid or self.args.flowrefinenorm):
			inputmaps = 212
			if(self.args.flowrefinenorm):
				inputmaps = 296
			self.conv_flow3 = nn.Sequential(
				nn.Conv2d(inputmaps, self.nf*4, [1, 1], 1, [0, 0]),
				nn.ReLU(),
				nn.Conv2d(self.nf*4, 4 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(4 * self.nf, 4 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
			)
		##########################################################

		if(self.args.pacFlow):
			self.pac_features = nn.Sequential(
				nn.Conv2d(3, 24, 3, 2, 1),
				nn.ReLU(),
				nn.Conv2d(24, 24, 3, 2, 1),
				nn.ReLU(),
				nn.Conv2d(24, 6, 3, 2, 1),
				nn.ReLU()
				)
			self.pac_features_whole = nn.Sequential(
				nn.Conv2d(12, 3, 3, 1, 1),
				nn.ReLU()
				)
			# PAC net
			self.conv_flow1 = MyConv2dPac(2*self.nf, self.nf, 3, 1, 1)
			self.conv_flow2 = nn.ModuleList([MyConv2dPac(2*self.nf + 4, 2 * self.nf, 3, 1, 1)])
			self.conv_flow2.append(nn.ReLU())
			self.conv_flow2.append(MyConv2dPac(2 * self.nf, 2 * self.nf, 3, 1, 1))
			self.conv_flow2.append(nn.ReLU())
			self.conv_flow2.append(MyConv2dPac(2 * self.nf, 2 * self.nf, 3, 1, 1))
			self.conv_flow2.append(nn.ReLU())
			# No PAC
			self.conv_flow2.append(nn.Conv2d(2 * self.nf, 2 * self.nf, 3, 1, 1))
			self.conv_flow2.append(nn.ReLU())
			self.conv_flow2.append(nn.Conv2d(2 * self.nf, 2 * self.nf, 3, 1, 1))
		elif(self.args.pacFlow2):
			# PAC net
			self.conv_flow1 = nn.Conv2d(2*self.nf, self.nf, 3, 1, 1)
			self.conv_flow2 = nn.ModuleList([nn.Conv2d(2*self.nf + 4, 2 * self.nf, 3, 1, 1)])
			self.conv_flow2.append(nn.ReLU())
			self.conv_flow2.append(MyConv2dPac(2 * self.nf, 2 * self.nf, 3, 1, 1))
			self.conv_flow2.append(nn.ReLU())
			self.conv_flow2.append(MyConv2dPac(2 * self.nf, 2 * self.nf, 3, 1, 1))
			self.conv_flow2.append(nn.ReLU())
			# No PAC
			self.conv_flow2.append(nn.Conv2d(2 * self.nf, 2 * self.nf, 3, 1, 1))
			self.conv_flow2.append(nn.ReLU())
			self.conv_flow2.append(nn.Conv2d(2 * self.nf, 2 * self.nf, 3, 1, 1))

			self.conv_flow3 = nn.ModuleList([nn.Conv2d( 4 + self.nf * 4, self.nf, [1, 1], 1, [0, 0])])
			self.conv_flow3.append(nn.ReLU())
			self.conv_flow3.append(MyConv2dPac(self.nf, 2 * self.nf,3,1,1))
			self.conv_flow3.append(nn.ReLU())
			self.conv_flow3.append(MyConv2dPac(2 * self.nf, 4 * self.nf,3,1,1))
			self.conv_flow3.append(
				nn.Sequential(
					nn.ReLU(),
					nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
					nn.ReLU(),
					nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
					nn.ReLU(),
					nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
				)
			)

		

		if(self.args.pacFlow3):
			self.conv_flow3_pac = nn.ModuleList([MyConv2dPac(4, 2 * self.nf,3,1,1)])
			self.conv_flow3_pac.append(nn.ReLu())
			self.conv_flow3_pac.append(MyConv2dPac(2 * self.nf, 4 * self.nf, 3, 1, 1))
				
			self.conv_flow3 = nn.Sequential(
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
			)

		##################### JUST ARCHTEST FROM HERE ON ###############################################
		if(self.args.archTest):
			self.conv_flow_bottom = nn.Sequential( 
				nn.Conv2d(2*self.args.dctvfi_nf * self.args.img_ch, 2*self.nf, [3,3], 1, [1,1]), # kernelsize,stride,padding
				nn.ReLU(),
				nn.Conv2d(2*self.nf, 2*self.nf, [3,3], 1, [1,1]), 
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 6, [3,3], 1, [1,1]), 
				)

			# BFlowNet
			self.conv_flow1 = nn.Conv2d(2*self.args.dctvfi_nf * self.args.img_ch, self.nf, [3, 3], 1, [1, 1])
			
			self.conv_flow2 = nn.Sequential(
				nn.Conv2d(2*self.nf + 4, 1 * self.nf, [3, 3], 1, [1,1]), 
				nn.ReLU(),
				nn.Conv2d(1 * self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(1 * self.nf, int(0.5 * self.nf), [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(int(0.5 * self.nf), int(0.5 * self.nf), [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(int(0.5 * self.nf), 6, [3, 3], 1, [1, 1]),
			)

			# TFlowNet
			self.conv_flow3 = nn.Sequential(
				nn.Conv2d(4 + self.args.dctvfi_nf * self.args.img_ch * 4, self.nf, [1, 1], 1, [0, 0]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(1 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(1 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
			)
		



		self.refine_unet = PCARefineUNet(args)
		res = sum(p.numel() for p in self.refine_unet.parameters())
		print("Parameters of refine UNET: ",res)
		self.lrelu = nn.ReLU()
		#########################				FORWARD ADAPTATIONS		##################################
		if self.args.raftupflowfeat:
			self.raftupflow = nn.Sequential(nn.Conv2d(48,64*9,3,1,1),nn.ReLU(inplace=True),nn.Conv2d(64*9,64*9,1,1,padding=0),nn.ReLU())

		if self.args.raftupflowimage:
			self.raftupflow = nn.Sequential(nn.Conv2d(3,9,3,2,1),nn.ReLU(inplace=True),nn.Conv2d(9,81,3,2,1),nn.ReLU(inplace=True),nn.Conv2d(81,256,3,2,1),nn.ReLU(inplace=True),nn.Conv2d(256,64*9,1,1,padding=0),nn.ReLU())
		if(self.args.pacupfor or self.args.oldpacupfor or self.args.nonadditivepac):
			self.endUp = nn.ModuleList([])
			for i in range(int(np.log2(self.args.scales[0]))): #-(1 if(self.args.pwcflowrefine)else 0)
				self.endUp.append(nn.ModuleList([MyConv2dPacTranspose(2,2,5,stride=2,padding=2,output_padding=1),nn.ReLU()]))
			self.pac_featuresFinalFlow = nn.Sequential(nn.Conv2d(3,3,3,1,1),nn.ReLU(),nn.Conv2d(3,3,3,1,1),nn.ReLU())
		if(self.args.teacherflowresid  ):
			self.conv_flow_teach = nn.Sequential(
				nn.Conv2d(17, 17, [1, 1], 1, [0, 0]),
				nn.ReLU(),
				nn.Conv2d(17, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(1 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(2 * self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				#nn.UpsamplingNearest2d(scale_factor=2),
				nn.Conv2d(1 * self.nf, self.nf, [3, 3], 1, [1, 1]),
				nn.ReLU(),
				nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
			)
			self.refine_unet_teach = PCARefineUNet(args,teach=True)

		if(self.args.forrefpacflow):
			self.forpacflow = nn.ModuleList([])
			self.forpacflow.append(nn.ModuleList([MyConv2dPac(2,4,3,1,1),nn.ReLU()]))
			self.forpacflow.append(nn.ModuleList([MyConv2dPac(4, 8,3,1,1),nn.ReLU()]))
			self.forpacflow.append(nn.ModuleList([MyConv2dPac(8, 2,3,1,1),nn.ReLU()]))
			self.feat_forpacflow = nn.Sequential(nn.Conv2d(3,12,3,2,1),nn.ReLU(),nn.Conv2d(12,12,3,1,1),nn.ReLU(),nn.UpsamplingNearest2d(scale_factor=2),nn.Conv2d(12,3,3,1,1),nn.ReLU())
		if(self.args.forrefRFBflow):
			self.preRFB_extrac = nn.Sequential(nn.Conv2d(6,12,3,2,1),nn.ReLU(),nn.Conv2d(12,24,3,2,1),nn.ReLU(),nn.UpsamplingNearest2d(scale_factor=4))
			self.flowRFB_extrac = BasicRFB(24+6+4,4,stride=1)
		if(self.args.ownoccl):
			self.alphas = torch.nn.Parameter(torch.ones((6),device=self.args.gpu).double())

		if(self.args.iteratpacup):
			self.PreLu = nn.PReLU()

			self.iterpacup = nn.ModuleList([])
			for i in range(3):
				self.iterpacup.append(nn.ModuleList([MyConv2dPac(2,2,3,stride=1,padding=1),self.PreLu]))
			self.iterpacfeats = nn.Sequential(nn.Conv2d(3,3,3,1,1),self.PreLu,nn.Conv2d(3,3,3,1,1),self.PreLu,nn.Conv2d(3,3,3,1,1),self.PreLu)

		if(self.args.simpleupsam):
			self.simple_upflow = nn.Sequential(nn.Conv2d(10,20,3,2,1),nn.ReLU(),nn.Conv2d(20,20,3,1,1),nn.ReLU(),nn.UpsamplingNearest2d(scale_factor=2),nn.Conv2d(20,4,5,1,2))
		#self.T_param =  torch.nn.Parameter(torch.ones((1),device=self.args.gpu).double())
		if(self.args.sminterp):
			self.T_param =  torch.nn.Parameter(torch.ones((1),device=self.args.gpu).double())
			self.T_param.requires_grad = False
		if(self.args.impmasksoftsplat):
			self.z_alpha =  torch.nn.Parameter(torch.ones((2),device=self.args.gpu).double())
		#######################			UPSAMPLING VARIANTS 		################################################
		if(self.args.upsamplingLayers ): #[8,4,12,12]
			self.levelUpsampling = nn.ConvTranspose2d(4,4,4,2,1) #kernel,stride,padding
			self.refUpsampling = nn.ConvTranspose2d(4,4,4,2,1)
		if(self.args.addUpsam):
			self.levelUpsampling = nn.Sequential(nn.ConvTranspose2d(4,4,4,2,1),nn.ReLU()) #kernel,stride,padding
			self.refUpsam = nn.ModuleList([])
			for i in range(int(np.log2(self.args.scales[0]))):
				self.refUpsam.append(nn.Sequential(nn.ConvTranspose2d(4,4,4,2,1),nn.ReLU()))
		if(self.args.pacUpsampling):
			self.pacUmsamplerPre = MyConv2dPacTranspose(2,2,5,stride=2,padding=2,output_padding=1)

		if(self.args.pacUpsamplingNew):
			self.pacUmsamplerPre = nn.ModuleList([MyConv2dPacTranspose(2,2,5,stride=2,padding=2,output_padding=1),nn.ReLU()])
			if(self.args.scales[0] == 8):
				self.pacFeatures = nn.Sequential(nn.Conv2d(3,3,3,2,1),nn.ReLU(),nn.Conv2d(3,3,3,2,1),nn.ReLU(),nn.Conv2d(3,3,3,2,1),nn.ReLU())
			elif(self.args.scales[0]==4):
				self.pacFeatures = nn.Sequential(nn.Conv2d(3,3,3,2,1),nn.ReLU(),nn.Conv2d(3,3,3,2,1),nn.ReLU())
			else:
				raise Exception("pacupsampling works only for scale 0 or 8 or 4")

			self.pac_featuresFinalFlow = nn.Sequential(nn.Conv2d(3,3,3,1,1),nn.ReLU(),nn.Conv2d(3,3,3,1,1),nn.ReLU())

			self.refUpsam = nn.ModuleList([])
			for i in range(int(np.log2(self.args.scales[0]))):
				self.refUpsam.append(nn.ModuleList([MyConv2dPacTranspose(2,2,5,stride=2,padding=2,output_padding=1),nn.ReLU()]))
			#self.pacFeat = nn.Sequential(nn.Conv2d(4,3,4,2,1),nn.ReLU())
			#self.pacUpsampleRef = nn.ModuleList([])
			#for i in range(int(math.log2(self.args.scales[0]))):
			#	self.pacUpsampleRef.append(PacConvTranspose2d(2,2,5,stride=2,padding=2,output_padding=1))

		if(self.args.zeroUpsamp or self.args.zeroUpsamp2):
			self.pacUpper = nn.ModuleList([])
			for i in range(self.args.S_trn+1): 
				self.pacUpper.append(MyConv2dPacTranspose(2,2,5,stride=2,padding=2,output_padding=1))

				
			self.impmaskUp = MyConv2dPacTranspose(1,1,5,stride=2,padding=2,output_padding=1)
													

			#self.pacUpsampleRef = nn.Sequential(*self.pacUpsampleRef)
		#vgg16 = torchvision.models.vgg16()
		#self.vgg16_conv = torch.nn.Sequential(*list(vgg.children())[0][:1])

		# input_big shape : [B,C*2*wiS*wiS,H,W]
		# input_small shape: [B,C*2*wiS_*wiS_,H_,W_]
	def forward(self, feat_x, flow_l_prev, t_value, level, is_training,normInput=0,validation=False,epoch=0,feat_pyr=[],mypwc=[],orig_images=None,frameT=None):
		'''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
		#B, C,  H, W = feat_x.size()
		B,C,H,W = feat_x.shape
		

		feat_x = feat_x.reshape(B,2,self.args.img_ch*self.args.dctvfi_nf,H,W)
		feat0_l = feat_x[:,0,:,:,:]
		feat1_l = feat_x[:,1,:,:,:]
		torch.cuda.empty_cache()
		x_l = normInput




		start_time = time.time()
		if self.args.justpwcflow:
			if(self.args.phase == "train" or level==0):
				scale_factor = 1/2 if(self.args.justpwcadaption)else 1/4
				imtem0 =x_l[:,:,0,:] if(self.args.lowresvers)else F.interpolate(x_l[:,:,0,:], scale_factor=scale_factor, mode='bilinear', align_corners=self.args.align_cornerse)
				imtem1 =x_l[:,:,1,:] if(self.args.lowresvers)else  F.interpolate(x_l[:,:,1,:], scale_factor=scale_factor, mode='bilinear', align_corners=self.args.align_cornerse)
				torch.cuda.empty_cache()

				with torch.no_grad():
					flow_l = self.mypwc.get_flow(imtem0,imtem1)

				#flow_l = F.interpolate(flow_l, scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
				#print("Level: ",level," = ",flow_l.shape, " |||",imtem0.shape)
				imtem1 = 0
				imtem0 = 0
			else:
				flow_l = 0
		else:
			## Flow estimation
			if flow_l_prev is None:
				if(self.args.pwcbottomflow):
					with torch.no_grad():
						flow_l_tmp = self.mypwc.get_flow(x_l[:,:,0,:],x_l[:,:,1,:])
					flow_l = flow_l_tmp[:,:4,:,:]
				elif self.args.additivePWC:
					imtem0 =  F.interpolate(x_l[:,:,0,:], scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
					imtem1 =  F.interpolate(x_l[:,:,1,:], scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
					with torch.no_grad():
						flow_pwc = self.mypwc.get_flow(imtem0,imtem1)
						flow_pwc = 0.5 * F.interpolate(flow_pwc, scale_factor=1/2, mode='bilinear', align_corners=self.args.align_cornerse)

					flow_l = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l,flow_pwc), dim=1))
					if(is_training):
						flow_l = flow_l[:,:4,:,:] + flow_pwc

				elif(self.args.addPWCOneFlow):
					flow_pwc = feat_pyr

					flow_l = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l,flow_pwc), dim=1))
					if(is_training):
						flow_l = flow_l[:,:4,:,:] + flow_pwc

				else:
					flow_l_tmp = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l), dim=1))
					flow_l = flow_l_tmp[:,:4,:,:]
			else:

				temp_shap = flow_l_prev.shape # [8,4,12,12] ||| feat1_l: [8,48,24,24]
				if(self.args.additivePWC):
					imtem0 =  F.interpolate(x_l[:,:,0,:], scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
					imtem1 =  F.interpolate(x_l[:,:,1,:], scale_factor=1/4, mode='bilinear', align_corners=self.args.align_cornerse)
					with torch.no_grad():
						flow_pwc = self.mypwc.get_flow(imtem0,imtem1)
						flow_pwc = 0.5 * F.interpolate(flow_pwc, scale_factor=1/2, mode='bilinear', align_corners=self.args.align_cornerse)
				elif(self.args.addPWCOneFlow):
					flow_pwc = feat_pyr

				if(self.args.upsamplingLayers and ((not is_training and not validation) or epoch>self.args.upsamplingEpoch)):
					#print("previous shape: ",flow_l_prev.shape,feat1_l.shape)
					up_flow_l_prev = self.levelUpsampling(flow_l_prev.detach()) 
				elif(self.args.addUpsam):
					up_flow_l_prev = F.interpolate(flow_l_prev.detach(), scale_factor=2, mode='bilinear', align_corners=self.args.align_cornerse)
					up_flow_l_prev += self.levelUpsampling(flow_l_prev.detach())

				elif(self.args.pacUpsamplingNew):
					up_flow_l_prev = F.interpolate(flow_l_prev.detach(), scale_factor=2, mode='bilinear', align_corners=self.args.align_cornerse)
					
					#guide0 = F.interpolate(normInput[:,:,0,:,:], size=(feat1_l.shape[2],feat1_l.shape[3]), mode='bilinear', align_corners=self.args.align_cornerse)
					#guide1 = F.interpolate(normInput[:,:,1,:,:], size=(feat1_l.shape[2],feat1_l.shape[3]), mode='bilinear', align_corners=self.args.align_cornerse)
					#print("GUIDE SHAPE: ",guide0.shape,normInput.shape)
					guide0 = self.pacFeatures(normInput[:,:,0,:,:])
					guide1 = self.pacFeatures(normInput[:,:,1,:,:])

					#print("pre ",self.pacUmsamplerPre)
					up_flow_01 = self.pacUmsamplerPre[1](self.pacUmsamplerPre[0](flow_l_prev[:,:2,:,:].detach(),guide0))
					up_flow_10 = self.pacUmsamplerPre[1](self.pacUmsamplerPre[0](flow_l_prev[:,2:,:,:].detach(),guide1))
					up_flow_l_prev += torch.cat([up_flow_01,up_flow_10],dim=1)

				elif(self.args.pacUpsampling and ((not is_training and not validation) or epoch>self.args.upsamplingEpoch)):
					guide0 = F.interpolate(normInput[:,:,0,:,:], size=(feat1_l.shape[2],feat1_l.shape[3]), mode='bilinear', align_corners=self.args.align_cornerse)
					guide1 = F.interpolate(normInput[:,:,1,:,:], size=(feat1_l.shape[2],feat1_l.shape[3]), mode='bilinear', align_corners=self.args.align_cornerse)
					#print("GUIDE SHAPE: ",guide0.shape,normInput.shape)
					up_flow_01 = self.pacUmsamplerPre(flow_l_prev[:,:2,:,:].detach(),guide0)
					up_flow_10 = self.pacUmsamplerPre(flow_l_prev[:,2:,:,:].detach(),guide1)
					up_flow_l_prev = torch.cat([up_flow_01,up_flow_10],dim=1)
				elif(self.args.zeroUpsamp):
					#assert feat1_l.shape[2] == flow_l_prev.shape[2]*2
					up_flow_l_prev = F.interpolate(flow_l_prev.detach(), scale_factor=2, mode='bilinear', align_corners=self.args.align_cornerse)
					up_flow_l_prev[:,:2,:,:] += self.lrelu(self.pacUpper[level](flow_l_prev[:,:2,:,:].detach(),feat_pyr[level][0]))
					up_flow_l_prev[:,2:,:,:] += self.lrelu(self.pacUpper[level](flow_l_prev[:,2:,:,:].detach(),feat_pyr[level][1]))
				elif(self.args.zeroUpsamp2):
					up_flow_l_prev = F.interpolate(flow_l_prev.detach(), scale_factor=2, mode='bilinear', align_corners=self.args.align_cornerse)
					up_flow_l_prev[:,:2,:,:] += self.lrelu(self.pacUpper[level](flow_l_prev[:,:2,:,:].detach(),feat_pyr[level][0]))
					up_flow_l_prev[:,2:,:,:] += self.lrelu(self.pacUpper[level](flow_l_prev[:,2:,:,:].detach(),feat_pyr[level][1]))
				else:
					#print("AHA ES IST VALIDATION DESWEGEN BIN ICH HIER oder sonst was")
					up_flow_l_prev = F.interpolate(flow_l_prev.detach(), size=(feat1_l.shape[2],feat1_l.shape[3]), mode='bilinear', align_corners=self.args.align_cornerse)
				#print("multiplied: ",up_flow_l_prev.shape[3]/temp_shap[3])
				#print("upscaled flow: ",up_flow_l_prev.shape)
				up_flow_l_prev *= (up_flow_l_prev.shape[3]/temp_shap[3])
				if(self.args.softsplat):
					# :2 1 -> 0  ## 2: 0 -> 1
					# flow_10_l = flow_l[:,:2,:,:]
					# flow_01_l = flow_l[:,2:,:,:]
					if(self.args.additivePWC or self.args.addPWCOneFlow):
						up_flow_l_prev += flow_pwc
					warped_feat1_l = self.softsplat(feat1_l, up_flow_l_prev[:,:2,:,:])
					warped_feat0_l = self.softsplat(feat0_l, up_flow_l_prev[:,2:,:,:])
				else:
					warped_feat1_l = self.bwarp(feat1_l, up_flow_l_prev[:,:2,:,:],withmask=not self.args.maskLess)
					warped_feat0_l = self.bwarp(feat0_l, up_flow_l_prev[:,2:,:,:],withmask=not self.args.maskLess)

				
				##################################################		UPSAMPLING OF LAST FLOW 	##########################################################

				
				if(self.args.pacFlow ):
					# Guiding Features each
					guide0 = self.pac_features(normInput[:,:,0,:,:])
					guide1 = self.pac_features(normInput[:,:,1,:,:])
					# Guiding Features for conv_flow2
					both_guide =self.pac_features_whole(torch.cat([guide0,guide1],dim=1))
					# conv_flow1
					out0 = self.conv_flow1(torch.cat([feat0_l, warped_feat1_l],dim=1),guide0)
					out1 = self.conv_flow1(torch.cat([feat1_l, warped_feat0_l],dim=1),guide1)
					# Input to convflow2
					catted = torch.cat([out0,out1,up_flow_l_prev],dim=1)
					abwechsler = 0
					for layer in self.conv_flow2:
						if(abwechsler % 2 == 0 and abwechsler<6):
							catted = layer(catted,both_guide)
							abwechsler += 1
						else:
							catted = layer(catted)
							abwechsler += 1
					flow_l_tmp = catted
				elif(self.args.pacFlow2):
					# Guiding Features each
					#guide0 = self.pac_features(normInput[:,:,0,:,:])
					#guide1 = self.pac_features(normInput[:,:,1,:,:])
					# Guiding Features for conv_flow2
					#both_guide =self.pac_features_whole(torch.cat([guide0,guide1],dim=1))
					# conv_flow1
					out0 = self.conv_flow1(torch.cat([feat0_l, warped_feat1_l],dim=1))
					out1 = self.conv_flow1(torch.cat([feat1_l, warped_feat0_l],dim=1))
					# Input to convflow2
					catted = torch.cat([out0,out1,up_flow_l_prev],dim=1)
					abwechsler = 0

					guide = self.conv_flow2[0](catted) # conv2d
					guide = self.conv_flow2[1](guide)	# relu
					catted = self.conv_flow2[2](guide,guide)	# Pacconv
					catted = self.conv_flow2[3](catted)	# relu
					catted = self.conv_flow2[4](catted,guide)	# Pacconv
					catted = self.conv_flow2[5](catted)	# relu
					catted = self.conv_flow2[6](catted)	# conv2d
					catted = self.conv_flow2[7](catted)	# relu
					catted = self.conv_flow2[8](catted)	# conv2d

					flow_l_tmp = catted
				else:
					# print("before convflow2")
					# time.sleep(3)
					# :2 1 -> 0  ## 2: 0 -> 1
					flow_l_tmp = self.conv_flow2(torch.cat([self.conv_flow1(torch.cat([feat0_l, warped_feat1_l],dim=1)), self.conv_flow1(torch.cat([feat1_l, warped_feat0_l],dim=1)), up_flow_l_prev],dim=1))
					# print("after conv_flow2")
					# time.sleep(3)
					#torch_prints(flow_l_tmp,"flow other level")
				# W_padding = up_flow_l_prev.shape[3] - flow_l_tmp.shape[3]
				# H_padding = up_flow_l_prev.shape[2] - flow_l_tmp.shape[2]
				# if((abs(W_padding)>0 or abs(H_padding) >0) ):#and is_training
				# 	raise Exception("Padding is non zero: ",W_padding," ",H_padding) 
				# flow_l_tmp = F.pad(flow_l_tmp, (0, W_padding, 0,H_padding), self.args.flow_padding)
				if((self.args.additivePWC or self.args.addPWCOneFlow) and not is_training):
					flow_l = flow_l_tmp[:,:4,:,:] + up_flow_l_prev - flow_pwc
				else:
					flow_l = flow_l_tmp[:,:4,:,:] + up_flow_l_prev 
##############################################################################################################################


		# if(self.args.timetest and level == 0):
		# 	print("Level 0 01 10 flow: ",time.time()-start_time)
		# 	start_time = time.time()
		# from here on common part including coarsest scale ##############################################
		#########################################		REFINE FLOW 		##################################################################
		if(not self.args.cutoffUnnec):
			z_01_l = torch.sigmoid(flow_l_tmp[:,4:5,:,:])
			z_10_l = torch.sigmoid(flow_l_tmp[:,5:6,:,:])
		
		if not is_training and level!=0: 
			return flow_l#,[z_01_l,z_10_l] 
		
		# print("before level 0 goes on")
		# time.sleep(3)	
		# BIOF-T
		if(self.args.softsplat):
			flow_10_l = flow_l[:,:2,:,:]
			flow_01_l = flow_l[:,2:,:,:]
		else:
			flow_01_l = flow_l[:,:2,:,:]
			flow_10_l = flow_l[:,2:,:,:]
		

		if(not self.args.softsplat):
			
			## Complementary Flow Reversal (CFR)
			flow_forward, norm0_l = self.z_fwarp(flow_01_l, t_value * flow_01_l, z_01_l)  ## Actually, F (t) -> (t+1). Translation only. Not normalized yet
			flow_backward, norm1_l = self.z_fwarp(flow_10_l, (1-t_value) * flow_10_l, z_10_l)  ## Actually, F (1-t) -> (-t). Translation only. Not normalized yet
			
			
			flow_t0_l = -(1-t_value) * ((t_value)*flow_forward) + (t_value) * ((t_value)*flow_backward) # The numerator of Eq.(1) in the paper.
			flow_t1_l = (1-t_value) * ((1-t_value)*flow_forward) - (t_value) * ((1-t_value)*flow_backward) # The numerator of Eq.(2) in the paper.
			
			norm_l = (1-t_value)*norm0_l + t_value*norm1_l
			mask_ = (norm_l.detach() > 0).type(norm_l.type())
			flow_t0_l = (1-mask_) * flow_t0_l + mask_ * (flow_t0_l.clone() / (norm_l.clone() + (1-mask_))) # Divide the numerator with denominator in Eq.(1)
			flow_t1_l = (1-mask_) * flow_t1_l + mask_ * (flow_t1_l.clone() / (norm_l.clone() + (1-mask_))) # Divide the numerator with denominator in Eq.(2)

			#print("Before error: ",feat0_l.shape,flow_t0_l.shape)
			## Feature warping
			warped0_l = self.bwarp(feat0_l, flow_t0_l,withmask=not self.args.maskLess)
			warped1_l = self.bwarp(feat1_l, flow_t1_l,withmask=not self.args.maskLess)

			## Flow refinement TFlownet
			if(self.args.ignoreFeatx):
				flow_refine_l = torch.cat([ flow_t0_l, flow_t1_l], dim=1)
			else:
				flow_refine_l = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l, flow_t0_l, flow_t1_l], dim=1)

			if(self.args.pacFlow2):
				flow_refine_l = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l, flow_t0_l, flow_t1_l], dim=1)
				#guide = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l], dim=1)
				guide = self.conv_flow3[0](flow_refine_l)	# Conv2d
				guide = self.conv_flow3[1](guide)			# relu
				flow_refine_l = self.conv_flow3[2](guide,guide) # Pacconv2d
				flow_refine_l = self.conv_flow3[3](flow_refine_l)	# Relu
				flow_refine_l = self.conv_flow3[4](flow_refine_l,guide)	# Pacconv2d
				flow_refine_l = self.conv_flow3[5](flow_refine_l)	# Rest
			elif(self.args.pacFlow3):
				flow_refine_l = torch.cat([ flow_t0_l, flow_t1_l], dim=1)
				guide = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l], dim=1)
				flow_refine_l = self.conv_flow3_pac[0](flow_refine_l,guide)
				flow_refine_l = self.conv_flow3_pac[1](flow_refine_l)
				flow_refine_l = self.conv_flow3_pac[2](flow_refine_l,guide)
				flow_refine_l = self.conv_flow3(flow_refine_l)
			else:
				flow_refine_l = self.conv_flow3(flow_refine_l)
		
			# MY PADDING
			W_padding = flow_t0_l.shape[3] - flow_refine_l.shape[3]
			H_padding = flow_t0_l.shape[2] - flow_refine_l.shape[2]
			#print("Padding done 2: ", W_padding,H_padding)
			if((abs(W_padding)>0 or abs(H_padding) >0)):
				raise Exception("Padding is non zero: ",W_padding," ",H_padding) 
			flow_refine_l = F.pad(flow_refine_l, (0, W_padding, 0,H_padding), self.args.flow_padding)
		elif(self.args.pwcflowrefine):
			dsfactor = 1/2
			if(self.args.justpwcadaption):
				dsfactor = 1/4
			tem01 = dsfactor* F.interpolate(flow_01_l,scale_factor=dsfactor,mode="bilinear",align_corners=self.args.align_cornerse)
			tem10 = dsfactor* F.interpolate(flow_10_l,scale_factor=dsfactor,mode="bilinear",align_corners=self.args.align_cornerse)
			flowback_0 = self.bwarp(tem10*t_value, (1 -t_value)*tem01,withmask=not self.args.outMaskLess) # t->0
			flowback_1 = self.bwarp(tem01*(1 -t_value), t_value*tem10,withmask=not self.args.outMaskLess) # t -> 1
			if(self.args.lowresvers):
				feat0_l = F.interpolate(feat0_l,scale_factor=4,mode="nearest")
				feat1_l = F.interpolate(feat1_l,scale_factor=4,mode="nearest")
			
			warped0_l = self.softsplat(feat0_l, tem01*(t_value))
			warped1_l = self.softsplat(feat1_l, tem10*(1 -t_value))
			flow_refine_l = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l, flowback_0, flowback_1,tem01,tem10], dim=1)
			flow_refine_l = self.conv_flow3(flow_refine_l)
			tem01 += flow_refine_l[:, :2, :, :]
			tem10 += flow_refine_l[:, 2:4, :, :]

			flow_01_l = tem01
			flow_10_l = tem10


			flow_t0_l = t_value.view(-1,1,1,1) * flow_01_l
			flow_t1_l = (1 - t_value).view(-1,1,1,1) * flow_10_l
			if(self.args.phase != "test" or self.args.testgetflowout):
				flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)
			tem01 = 0
			tem10 = 0
		elif self.args.pwcresid:
			tem01 = flow_01_l 
			tem10 = flow_10_l 
			im0d = F.interpolate(x_l[:,:,0,:],scale_factor=1/8,mode="bilinear")
			im1d = F.interpolate(x_l[:,:,1,:],scale_factor=1/8,mode="bilinear")
			flowback_0 = self.bwarp(tem10*t_value, (1 -t_value)*tem01,withmask=not self.args.outMaskLess) # t->0
			flowback_1 = self.bwarp(tem01*(1 -t_value), t_value*tem10,withmask=not self.args.outMaskLess) # t -> 1

			im0d_back = self.bwarp(im0d,flowback_0,withmask=not self.args.outMaskLess)
			im1d_back = self.bwarp(im1d,flowback_1,withmask=not self.args.outMaskLess)
			im0d_forw = self.softsplat(im0d,t_value*tem01)
			im1d_forw = self.softsplat(im1d,(1-t_value)*tem10)

			with torch.no_grad():
				flowresid_back = self.mypwc.get_flow(im0d_back,im1d_back)
				flowresid_for = self.mypwc.get_flow(im0d_forw,im1d_forw)
			
			
			warped0_l = self.softsplat(feat0_l, tem01*(t_value))
			warped1_l = self.softsplat(feat1_l, tem10*(1 -t_value))
			flow_refine_l = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l, flowback_0, flowback_1,tem01,tem10,im0d_back,im1d_back,im0d_forw,im1d_forw], dim=1)
			flow_refine_l = self.conv_flow3(flow_refine_l)
			tem01 += flow_refine_l[:, :2, :, :]
			tem10 += flow_refine_l[:, 2:4, :, :]

			flow_01_l = tem01
			flow_10_l = tem10


			flow_t0_l = t_value.view(-1,1,1,1) * flow_01_l
			flow_t1_l = (1 - t_value).view(-1,1,1,1) * flow_10_l
			if(self.args.phase != "test" or self.args.testgetflowout):
				flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)
			tem01 = 0
			tem10 = 0
			im0d = 0
			im1d = 0
			im0d_back= 0
			im1d_back= 0
			im0d_forw= 0
			im1d_forw= 0
		elif self.args.flowrefinenorm:

			#flow_01_l = 2* F.interpolate(flow_01_l,scale_factor=2,mode="bilinear",align_corners=self.args.align_cornerse)
			#flow_10_l = 2* F.interpolate(flow_10_l,scale_factor=2,mode="bilinear",align_corners=self.args.align_cornerse)
			flowback_0 = self.bwarp(flow_10_l*t_value, (1 -t_value)*flow_01_l,withmask=not self.args.outMaskLess) # t->0
			flowback_1 = self.bwarp(flow_01_l*(1 -t_value), t_value*flow_10_l,withmask=not self.args.outMaskLess) # t -> 1

			
			warped0_l = self.softsplat(feat0_l, flow_01_l*(t_value))
			warped1_l = self.softsplat(feat1_l, flow_10_l*(1 -t_value))
			warped0_l_back = self.bwarp(feat0_l,flowback_0,withmask=not self.args.outMaskLess)
			warped1_l_back = self.bwarp(feat1_l,flowback_1,withmask=not self.args.outMaskLess)

			flow_refine_l = torch.cat([feat0_l, feat1_l, warped0_l, warped1_l, warped0_l_back,warped1_l_back,flowback_0, flowback_1,flow_01_l,flow_10_l], dim=1)
			flow_refine_l = self.conv_flow3(flow_refine_l)
			flow_01_l += flow_refine_l[:, :2, :, :]
			flow_10_l += flow_refine_l[:, 2:4, :, :]



			flow_t0_l = t_value.view(-1,1,1,1) * flow_01_l
			flow_t1_l = (1 - t_value).view(-1,1,1,1) * flow_10_l
			if(self.args.phase != "test" or self.args.testgetflowout):
				flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)
			tem01 = 0
			tem10 = 0
		else: # SIMPLE SOFTSPLAT###################################################################
			flow_t0_l = t_value.view(-1,1,1,1) * flow_01_l
			flow_t1_l = (1 - t_value).view(-1,1,1,1) * flow_10_l
			if(self.args.phase != "test" or self.args.testgetflowout):
				flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)

			if(self.args.dsstart>1 and level == 0):
				x_l = orig_images



		### Add Flow from BiOF-I(after CFR) to TFlownet-output
		if(not self.args.softsplat):
			flow_refine_l +=  torch.cat([flow_t0_l, flow_t1_l], dim=1)
			# The two flow directions
			flow_t0_l = flow_refine_l[:, :2, :, :]
			flow_t1_l = flow_refine_l[:, 2:4, :, :]

			warped0_l = self.bwarp(feat0_l, flow_t0_l,withmask=not self.args.maskLess)
			warped1_l = self.bwarp(feat1_l, flow_t1_l,withmask=not self.args.maskLess)

		## Flow upscale NOT NEEDED AS NO DOWNSCALING IS HAPPENING #####################################################################################
		#flow_t0_l = self.scale * F.interpolate(flow_t0_l, scale_factor=(self.scale, self.scale), mode='bilinear',align_corners=self.args.align_cornerse)
		#flow_t1_l = self.scale * F.interpolate(flow_t1_l, scale_factor=(self.scale, self.scale), mode='bilinear',align_corners=self.args.align_cornerse)

		######################################### 		Image warping and blending		#####################################
		if(self.args.norm_image_warp or self.args.pcanet):
			# norminputs: 8/8, 8/16, 8/32 
			if(self.args.ds_normInput):
				upscaleDict = [self.args.scales[0]/1.0 for i in self.args.scales] #{"0": 1.0, "1":2.0, "2":4.0, "3":8.0}
			else:
				upscaleDict = [i/1.0 for i in self.args.scales] #{"0": 4.0, "1":8.0, "2":16.0, "3":32.0}
			festerwert = upscaleDict[level]#20.0 if(level ==1) else 8.0     
			upscale =   x_l.shape[3]/flow_t0_l.shape[2]  #festerwert if(is_training) else
			#print("shape: ",x_l.shape,flow_t0_l.shape)
			if(not upscale.is_integer()):
				raise Exception("upscale factor is no integer!!! Upscale factor: " + str(upscale))
			upscale = int(upscale)
			#print("upscales factor: ",upscale)
			teach_flow_res = 0
			if(upscale < 1.1 and upscale >0.9):
				pass
			else:
				if(self.args.pacUpsamplingNew):
					temp0 =  upscale * F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					temp1 = upscale * F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					warpedim0temp = self.bwarp(x_l[:,:,0,:,:], temp0)
					warpedim1temp = self.bwarp(x_l[:,:,1,:,:], temp1)
					flow0_temp = flow_t0_l
					flow1_temp = flow_t1_l
					guide0 = self.pac_featuresFinalFlow(warpedim0temp)
					guide1 = self.pac_featuresFinalFlow(warpedim1temp)
					warpedim1temp = 0
					warpedim0temp = 0
					temp1 = 0
					temp0 = 0
					torch.cuda.empty_cache()

					for layer in self.refUpsam:
						
						guide0temp = F.interpolate(guide0,scale_factor=((flow0_temp.shape[-1]*2)/guide0.shape[-1]),mode="bilinear",align_corners=self.args.align_cornerse)
						guide1temp = F.interpolate(guide1,scale_factor=((flow0_temp.shape[-1]*2)/guide0.shape[-1]),mode="bilinear",align_corners=self.args.align_cornerse)
						#print(layer,flow0_temp.shape,guide0temp.shape)
						flow0_temp = layer[1](layer[0](flow0_temp,guide0temp))
						flow1_temp = layer[1](layer[0](flow1_temp,guide1temp))


					flow_t0_l =  upscale * (flow0_temp + F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse))
					flow_t1_l = upscale * (flow1_temp + F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse))
					guide1temp = 0
					guide0temp = 0
					flow1_temp = 0
					flow0_temp = 0
					guide0 = 0
					guide1 = 0
					torch.cuda.empty_cache()
				elif(self.args.addUpsam):
					catInp = torch.cat([flow_t0_l,flow_t1_l],dim=1)
					for layer in self.refUpsam:
						catInp = layer(catInp)

					flow_t0_l =  upscale * (catInp[:,:2] + F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse))
					flow_t1_l = upscale * (catInp[:,2:] + F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse))
				elif(self.args.upsamplingLayers and ((not is_training and not validation) or epoch>self.args.upsamplingEpoch)):
					#print(flow_t0_l.shape)
					temp_flow = self.refUpsampling(torch.cat([flow_t0_l,flow_t1_l],dim=1))
						
					for i in range(int(math.log2(upscale))-1):
						temp_flow = self.refUpsampling(temp_flow)
					flow_t0_l = upscale * temp_flow[:,:2]
					flow_t1_l = upscale * temp_flow[:,2:]
				elif(self.args.pacUpsampling and False and ((not is_training and not validation) or epoch>self.args.upsamplingEpoch)):

					for i in range(int(math.log2(upscale))):
						temp_fac = self.args.scales[0]//(2**(i+1))
						if(temp_fac == 1):
							guide0 = normInput[:,:,0,:,:]
							guide1 = normInput[:,:,1,:,:]
						else:
							guide0 = F.interpolate(normInput[:,:,0,:,:], scale_factor=1/temp_fac, mode='bilinear', align_corners=self.args.align_cornerse)
							guide1 = F.interpolate(normInput[:,:,1,:,:],  scale_factor=1/temp_fac, mode='bilinear', align_corners=self.args.align_cornerse)
						flow_t0_l = self.pacUpsampleRef[i](flow_t0_l,guide0)
						flow_t1_l = self.pacUpsampleRef[i](flow_t1_l,guide1)

				elif(self.args.pacupfor):
					# detach as upsampled flow is added anyway!
					torch.cuda.empty_cache()
					flowUp_0t = flow_t0_l.detach().clone()
					flowUp_1t = flow_t1_l.detach().clone()
					for index,layer in enumerate(self.endUp):
						guide0 = x_l[:,:,0,:] if(index == 2)else F.interpolate(x_l[:,:,0,:], scale_factor=(1/((2-index)*2), 1/((2-index)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide1 = x_l[:,:,1,:] if(index == 2)else F.interpolate(x_l[:,:,1,:], scale_factor=(1/((2-index)*2), 1/((2-index)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide0 = self.pac_featuresFinalFlow(guide0) + guide0
						guide1 = self.pac_featuresFinalFlow(guide1) + guide1
						#print("before in:" ,flowUp_0t.shape,guide0.shape)
						flowUp_0t = self.endUp[index][0](flowUp_0t,guide0)*2
						flowUp_0t = self.endUp[index][1](flowUp_0t)
						flowUp_1t = self.endUp[index][0](flowUp_1t,guide1)*2
						flowUp_1t = self.endUp[index][1](flowUp_1t)

					flow_t0_l = flowUp_0t + upscale * F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t1_l = flowUp_1t +  upscale * F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t0_l *= upscale
					flow_t1_l *= upscale

				
					flow_10_l = flow_t1_l/(1-t_value)
					flow_01_l = flow_t0_l/(t_value)
					flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)
					
					
				elif(self.args.oldpacupfor):
					torch.cuda.empty_cache()
					flowUp_0t = flow_t0_l.detach().clone()
					flowUp_1t = flow_t1_l.detach().clone()
					for index,layer in enumerate(self.endUp):
						guide0 = x_l[:,:,0,:] if(index == 2)else F.interpolate(x_l[:,:,0,:], scale_factor=(1/((2-index)*2), 1/((2-index)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide1 = x_l[:,:,1,:] if(index == 2)else F.interpolate(x_l[:,:,1,:], scale_factor=(1/((2-index)*2), 1/((2-index)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide0 = self.pac_featuresFinalFlow(guide0) + guide0
						guide1 = self.pac_featuresFinalFlow(guide1) + guide1
						#print("before in:" ,flowUp_0t.shape,guide0.shape)
						flowUp_0t = self.endUp[index][0](flowUp_0t,guide0)
						flowUp_0t = self.endUp[index][1](flowUp_0t)
						flowUp_1t = self.endUp[index][0](flowUp_1t,guide1)
						flowUp_1t = self.endUp[index][1](flowUp_1t)
					flow_t0_l = flowUp_0t + F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t1_l = flowUp_1t +  F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t0_l *= upscale
					flow_t1_l *= upscale
				
					flow_10_l = flow_t1_l/(1-t_value)
					flow_01_l = flow_t0_l/(t_value)
					flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)

				elif(self.args.nonadditivepac):
					torch.cuda.empty_cache()
					flowUp_0t = flow_t0_l#.clone()#.detach().clone()
					flowUp_1t = flow_t1_l#.clone()#.detach().clone()

					for index,layer in enumerate(self.endUp):
						guide0 = x_l[:,:,0,:] if(index == 2)else F.interpolate(x_l[:,:,0,:], scale_factor=(1/((2-index)*2), 1/((2-index)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide1 = x_l[:,:,1,:] if(index == 2)else F.interpolate(x_l[:,:,1,:], scale_factor=(1/((2-index)*2), 1/((2-index)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide0 = self.pac_featuresFinalFlow(guide0) + guide0
						guide1 = self.pac_featuresFinalFlow(guide1) + guide1
						#print("before in:" ,flowUp_0t.shape,guide0.shape)
						flowUp_0t = self.endUp[index][0](flowUp_0t,guide0)
						flowUp_0t = self.endUp[index][1](flowUp_0t)
						flowUp_1t = self.endUp[index][0](flowUp_1t,guide1)
						flowUp_1t = self.endUp[index][1](flowUp_1t)
					flow_t0_l = flowUp_0t.clone() #+ F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t1_l = flowUp_1t.clone() #+  F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t0_l *= upscale
					flow_t1_l *= upscale
				
					flow_10_l = flow_t1_l/(1-t_value)
					flow_01_l = flow_t0_l/(t_value)
					flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)

				elif(self.args.iteratpacup):
					torch.cuda.empty_cache()
					flow_10_l = upscale * F.interpolate(flow_10_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_01_l = upscale * F.interpolate(flow_01_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)

					#flow_t0_l_tem = flow_t0_l#.clone()
					#flow_t1_l_tem = flow_t1_l#.clone()

					for iters in range(int(np.log2(upscale))):
						torch.cuda.empty_cache()
						assert int(np.log2(upscale)) == 3
						flow_t0_l = 2 * F.interpolate(flow_t0_l, scale_factor=(2, 2), mode='bilinear',align_corners=self.args.align_cornerse)
						flow_t1_l = 2 * F.interpolate(flow_t1_l, scale_factor=(2, 2), mode='bilinear',align_corners=self.args.align_cornerse)
						guide0 = x_l[:,:,0,:] if(iters == 2)else F.interpolate(x_l[:,:,0,:], scale_factor=(1/((2-iters)*2), 1/((2-iters)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide1 = x_l[:,:,1,:] if(iters == 2)else F.interpolate(x_l[:,:,1,:], scale_factor=(1/((2-iters)*2), 1/((2-iters)*2)), mode='bilinear',align_corners=self.args.align_cornerse)
						guide0 = self.iterpacfeats(guide0)
						guide1 = self.iterpacfeats(guide1)
						for index,layer in enumerate(self.iterpacup):
							#print(flow_t0_l_tem.shape,guide0.shape)
							flow_t0_l = self.iterpacup[index][0](flow_t0_l,guide0)
							flow_t0_l = self.iterpacup[index][1](flow_t0_l)
							flow_t1_l = self.iterpacup[index][0](flow_t1_l,guide1)
							flow_t1_l = self.iterpacup[index][1](flow_t1_l)


					#flow_t0_l =  upscale * F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					#flow_t1_l = upscale * F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					#flow_t0_l = flow_t0_l_tem
					#flow_t1_l = flow_t1_l_tem

					#self.PreLu
					flow_t0_l_tem = 0
					flow_t1_l_tem = 0
					guide0 = 0
					guide1 = 0
					torch.cuda.empty_cache()
					flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)
					#small network, PReLu, first upsample with transpose conv, than finetune. Use a little bit of featextrac for guide!
				elif(self.args.simpleupsam):
					flow_10_l = upscale * F.interpolate(flow_10_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_01_l = upscale * F.interpolate(flow_01_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					temflowref = torch.cat([flow_01_l,flow_10_l,x_l[:,:,1,:],x_l[:,:,0,:]],dim=1)
					temflowref = self.simple_upflow(temflowref)
					flow_01_l += temflowref[:,:2,:]
					flow_10_l += temflowref[:,2:,:]
					flow_t0_l = t_value * flow_01_l
					flow_t1_l = (1 -t_value) * flow_10_l
				elif self.args.raftupflowfeat and (not self.args.raftlevel0 or level==0):
					# 0_1
					mask_raft = self.raftupflow(feat0_l)
					N,_,HF,WF = flow_01_l.shape
					mask_raft = mask_raft.view(N, 1, 9, 8, 8, HF, WF)
					mask_raft = torch.softmax(mask_raft, dim=2)

					flow_t0_l_up = F.unfold(8 * flow_01_l, [3,3], padding=1)
					flow_t0_l_up = flow_t0_l_up.view(N, 2, 9, 1, 1, HF, WF)

					flow_t0_l_up = torch.sum(mask_raft * flow_t0_l_up, dim=2)
					flow_t0_l_up = flow_t0_l_up.permute(0, 1, 4, 2, 5, 3)
					
					flow_01_l = flow_t0_l_up.reshape(N, 2, 8*HF, 8*WF)
					# 1_0
					mask_raft = self.raftupflow(feat1_l)
					N,_,HF,WF = flow_10_l.shape
					mask_raft = mask_raft.view(N, 1, 9, 8, 8, HF, WF)
					mask_raft = torch.softmax(mask_raft, dim=2)

					flow_t0_l_up = F.unfold(8 * flow_10_l, [3,3], padding=1)
					flow_t0_l_up = flow_t0_l_up.view(N, 2, 9, 1, 1, HF, WF)

					flow_t0_l_up = torch.sum(mask_raft * flow_t0_l_up, dim=2)
					flow_t0_l_up = flow_t0_l_up.permute(0, 1, 4, 2, 5, 3)
					
					flow_10_l = flow_t0_l_up.reshape(N, 2, 8*HF, 8*WF)

					flow_t0_l = t_value * flow_01_l
					flow_t1_l = (1 -t_value) * flow_10_l
					# flow_t0_l =  upscale * F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					# flow_t1_l = upscale * F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					# flow_10_l = upscale * F.interpolate(flow_10_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					# flow_01_l = upscale * F.interpolate(flow_01_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
				elif self.args.raftupflowimage and (not self.args.raftlevel0 or level==0):
					# 0_1
					mask_raft = self.raftupflow(x_l[:,:,0,:])
					N,_,HF,WF = flow_01_l.shape
					mask_raft = mask_raft.view(N, 1, 9, 8, 8, HF, WF)
					mask_raft = torch.softmax(mask_raft, dim=2)

					flow_t0_l_up = F.unfold(8 * flow_01_l, [3,3], padding=1)
					flow_t0_l_up = flow_t0_l_up.view(N, 2, 9, 1, 1, HF, WF)

					flow_t0_l_up = torch.sum(mask_raft * flow_t0_l_up, dim=2)
					flow_t0_l_up = flow_t0_l_up.permute(0, 1, 4, 2, 5, 3)
					
					flow_01_l = flow_t0_l_up.reshape(N, 2, 8*HF, 8*WF)
					# 1_0
					mask_raft = self.raftupflow(x_l[:,:,1,:])
					N,_,HF,WF = flow_10_l.shape
					mask_raft = mask_raft.view(N, 1, 9, 8, 8, HF, WF)
					mask_raft = torch.softmax(mask_raft, dim=2)

					flow_t0_l_up = F.unfold(8 * flow_10_l, [3,3], padding=1)
					flow_t0_l_up = flow_t0_l_up.view(N, 2, 9, 1, 1, HF, WF)

					flow_t0_l_up = torch.sum(mask_raft * flow_t0_l_up, dim=2)
					flow_t0_l_up = flow_t0_l_up.permute(0, 1, 4, 2, 5, 3)
					
					flow_10_l = flow_t0_l_up.reshape(N, 2, 8*HF, 8*WF)

					flow_t0_l = t_value * flow_01_l
					flow_t1_l = (1 -t_value) * flow_10_l
				else:
					flow_t0_l =  upscale * F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_t1_l = upscale * F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_10_l = upscale * F.interpolate(flow_10_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					flow_01_l = upscale * F.interpolate(flow_01_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
					if(self.args.teacherflowresid and level== 0  and self.args.phase=="train" and not validation):
						flowback_0 = self.bwarp(flow_10_l*t_value, (1 -t_value)*flow_01_l,withmask=not self.args.outMaskLess) # t->0
						flowback_1 = self.bwarp(flow_01_l*(1 -t_value), t_value*flow_10_l,withmask=not self.args.outMaskLess) # t -> 1
						flow_refine_l_tem =  torch.cat([flow_t0_l, flow_t1_l], dim=1)
						teach_flow_res = self.conv_flow_teach(torch.cat([flow_refine_l_tem,x_l[:,:,0,:],x_l[:,:,1,:],frameT,flowback_0,flowback_1],dim=1))
						flow_refine_l_tem += teach_flow_res
						teach_flow_res = teach_flow_res.detach()
						flow_t0_l = flow_refine_l_tem[:,:2,:]
						flow_t1_l = flow_refine_l_tem[:,2:,:]
						flow_10_l = flow_t1_l/(1-t_value)
						flow_01_l = flow_t0_l/(t_value)

						flowback_0= 0
						flowback_1= 0
			

			if(is_training):
				tempshap = flow_t0_l.shape
				flow_t0_l = flow_t0_l[:,:,:self.args.patch_size,:self.args.patch_size]
				flow_t1_l = flow_t1_l[:,:,:self.args.patch_size,:self.args.patch_size]
				assert tempshap == flow_t0_l.shape, "tempshap and flow_t0_l not same"
		if(not is_training and not validation):
			#flow_t0_l = flow_t0_l[:,:,:self.output_size_test[0],:self.output_size_test[1]]
			#flow_t1_l = flow_t1_l[:,:,:self.output_size_test[0],:self.output_size_test[1]]
			#x_l = x_l[:,:,:,:self.output_size_test[0],:self.output_size_test[1]]
			torch.cuda.empty_cache()
		if(validation):
			flow_t0_l = flow_t0_l[:,:,:self.output_size_val[0],:self.output_size_val[1]]
			flow_t1_l = flow_t1_l[:,:,:self.output_size_val[0],:self.output_size_val[1]]
			x_l = x_l[:,:,:,:self.output_size_val[0],:self.output_size_val[1]]
		
		flowUp_0t = 0
		flowUp_1t = 0
		guide0 = 0
		guide1 = 0
		torch.cuda.empty_cache()


		# print("after flow upscale")
		# time.sleep(5)
		##################################		END WARPING AND MAYBE POST FLOW REFINEMENT 		#########################################

		#print("XL shape: ", x_l.shape,flow_t0_l.shape,upscale," level:",level)
		if(self.args.softsplat):
			#print("before softsplat: ",x_l[:,:,0,:,:].shape,flow_t0_l.shape)
			if(self.args.forrefpacflow):
				#inp_flow_ref = torch.cat([flow_t0_l, flow_t1_l], dim=1)
				flow_0t_temp = flow_t0_l.detach().clone()
				flow_1t_temp = flow_t1_l.detach().clone()
				guide0 = self.feat_forpacflow(x_l[:,:,0,:])
				guide1 = self.feat_forpacflow(x_l[:,:,1,:])
				for index,layer in enumerate(self.forpacflow):
					flow_0t_temp = self.forpacflow[index][0](flow_0t_temp,guide0)
					flow_0t_temp = self.forpacflow[index][1](flow_0t_temp)
					flow_1t_temp = self.forpacflow[index][0](flow_1t_temp,guide1)
					flow_1t_temp = self.forpacflow[index][1](flow_1t_temp)
				flow_t0_l += flow_0t_temp
				flow_t1_l += flow_1t_temp
			elif(self.args.forrefRFBflow):
				#flow_0t_temp = flow_t0_l.detach().clone()
				#flow_1t_temp = flow_t1_l.detach().clone()
				x_lshaped = x_l.reshape(x_l.shape[0],x_l.shape[1]*2,x_l.shape[3],x_l.shape[4])
				assert x_lshaped.shape[1] == 6, "two images so 6 color channel" 
				tem_feats = self.preRFB_extrac(x_lshaped)
				inp_refflow = torch.cat([flow_t0_l.detach().clone(),flow_t1_l.detach().clone(),tem_feats,x_lshaped], dim=1)
				ref_flow_addup = self.flowRFB_extrac(inp_refflow)
				flow_t0_l += ref_flow_addup[:,:2,:]
				flow_t1_l += ref_flow_addup[:,2:,:]



			if(self.args.impmasksoftsplat):
				assert self.z_alpha.requires_grad or self.args.TOptimization
				
				im_1_0 = self.bwarp(x_l[:,:,1,:,:], flow_01_l,withmask=not self.args.outMaskLess)
				z0 = torch.mean(self.z_alpha[0] * torch.abs(x_l[:,:,0,:,:] - im_1_0),dim=1,keepdim=True)

				im_0_1 = self.bwarp(x_l[:,:,0,:,:], flow_10_l,withmask=not self.args.outMaskLess)
				z1 = torch.mean(self.z_alpha[1] * torch.abs(x_l[:,:,1,:,:] - im_0_1),dim=1,keepdim=True)

				assert z0.get_device() == self.args.gpu and z1.get_device() == self.args.gpu
				warped_img0_l = self.softsplat(x_l[:,:,0,:,:], flow_t0_l,z=z0)
				warped_img1_l = self.softsplat(x_l[:,:,1,:,:], flow_t1_l,z=z1)
			else:
				warped_img0_l = self.softsplat(x_l[:,:,0,:,:], flow_t0_l)
				warped_img1_l = self.softsplat(x_l[:,:,1,:,:], flow_t1_l)

			z0 = 0
			z1 = 0
			im_1_0 = 0
			im_0_1 = 0
			torch.cuda.empty_cache()
			if(self.args.timetest):
				print("Rest before refinement: ",time.time()-start_time)
				start_time = time.time()
			# print("after input frame warping")
			# time.sleep(4)
			#torch_prints(warped_img0_l,"warped_img0_l")
			#torch_prints(warped_img1_l,"warped_img1_l")
			#torch_prints(x_l[:,:,0,:,:],"x_l")

			#print("warping xl to warped_img1_l: ",x_l[:,:,0,:,:].shape,warped_img0_l.shape)
		else:
			warped_img0_l = self.bwarp(x_l[:,:,0,:,:], flow_t0_l,withmask=not self.args.outMaskLess,minus=self.args.minus1bwarpEnd)
			warped_img1_l = self.bwarp(x_l[:,:,1,:,:], flow_t1_l,withmask=not self.args.outMaskLess,minus=self.args.minus1bwarpEnd)


		if(not self.args.feat_input):
			flow_0t_temp=0
			flow_1t_temp=0

			guide0=0
			guide1=0
			feat0_l =0
			feat1_l =0
			warped0_l = 0
			warped1_l = 0
			torch.cuda.empty_cache()
		



	
		torch.cuda.empty_cache()
		if(self.args.bothflowforsynth):
			flowback_0 = self.bwarp(flow_10_l, flow_01_l,withmask=not self.args.outMaskLess)
			flowback_1 = self.bwarp(flow_01_l, flow_10_l,withmask=not self.args.outMaskLess)
			im0_to1 =  self.bwarp(x_l[:,:,0,:,:], flow_10_l,withmask=not self.args.outMaskLess)
			im1_to0 =  self.bwarp(x_l[:,:,1,:,:], flow_01_l,withmask=not self.args.outMaskLess)
			inp_synth = torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l,flowback_0,flowback_1,im0_to1,im1_to0],dim=1)
			refine_out = self.refine_unet(inp_synth)
		elif(self.args.flowfromtto0 or self.args.flowfromtto0_lite):
			flowback_0 = self.bwarp(flow_10_l*t_value, (1 -t_value)*flow_01_l,withmask=not self.args.outMaskLess) # t->0
			flowback_1 = self.bwarp(flow_01_l*(1 -t_value), t_value*flow_10_l,withmask=not self.args.outMaskLess) # t -> 1
			if(not self.args.flowfromtto0_lite):
				im0_to1 =  self.bwarp(x_l[:,:,0,:,:], flow_10_l,withmask=not self.args.outMaskLess)
				im1_to0 =  self.bwarp(x_l[:,:,1,:,:], flow_01_l,withmask=not self.args.outMaskLess)
				inp_synth = torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l,flowback_0,flowback_1,im0_to1,im1_to0],dim=1)
			else:
				inp_synth = torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l,flowback_0,flowback_1],dim=1)
			refine_out = self.refine_unet(inp_synth)
		elif(self.args.bigallcomb or self.args.sminterp):
			flowback_0 = self.bwarp(flow_10_l*t_value, (1 -t_value)*flow_01_l,withmask=not self.args.outMaskLess) # t->0
			flowback_1 = self.bwarp(flow_01_l*(1 -t_value), t_value*flow_10_l,withmask=not self.args.outMaskLess) # t -> 1

			torch.cuda.empty_cache()
			if(self.args.interpOrigForw):
				refine_out = torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flowback_0,flowback_1],dim=1)
			else:
				im0_tot =  self.bwarp(x_l[:,:,0,:,:], flowback_0,withmask=not self.args.outMaskLess)
				im1_tot =  self.bwarp(x_l[:,:,1,:,:], flowback_1,withmask=not self.args.outMaskLess)
				refine_out = torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l,flowback_0,flowback_1,im0_tot,im1_tot],dim=1)
			

			flowback_1 = 0
			flowback_0 = 0
			#im0_tot = 0
			#im1_tot = 0
			if(self.args.phase == "test"):
				del flow_10_l  
				del flow_01_l  
				del flow_l  
				warped_feat1_l = 0  
				warped_feat0_l = 0  
				flow_l_tmp = 0 
				up_flow_l_prev = 0 
				del feat_x
				del feat0_l
				del feat1_l
				del flow_t0_l
				del flow_t1_l	
			torch.cuda.empty_cache()

			# print("before synthesis")
			#time.sleep(3)
			refine_out = self.refine_unet(refine_out)
			# print("after synthesis")
			# time.sleep(3)
			torch.cuda.empty_cache()
			if(self.args.timetest):	
				print("Refinement done: ",time.time()-start_time)
				start_time = time.time()
			
		else:
			refine_out = self.refine_unet(torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l],dim=1))
			#print("refine shape : ",refine_out.shape)
				
		feat0_l =0
		feat1_l =0
		warped0_l = 0
		warped1_l = 0

		if(self.args.sminterp):
			assert self.args.TOptimization == self.T_param.requires_grad

			if(self.args.sminterpInpIm and not self.args.interpOrigForw):
				num_softmax_combs = 6	
			else:
				num_softmax_combs = 4


			occ_all = F.softmax(refine_out[:, 0:num_softmax_combs, :, :]/self.T_param,dim=1)

			occ_0_l = occ_all[:,0:1,:,:]
			#occ_1_l = occ_all[:,1:2,:,:]
			#occ_all = nn.softmax(refine_out[:, 0:4, :, :]/self.T_param,dim=1)
		else:
			occ_0_l = torch.sigmoid(refine_out[:, 0:1, :, :])
			occ_1_l = 1-occ_0_l

		

		
			

		################################# 		IMAGE SYNTHESIS 			##########################################
		#print(occ_all.shape,warped_img0_l.shape,im0_tot.shape)
		error_res_teach = 0
		if(self.args.sminterp):
			divisor = ( (1-t_value).view(-1,1,1,1)*occ_all[:,0,:].unsqueeze(1) + t_value.view(-1,1,1,1)*occ_all[:,1,:].unsqueeze(1) + (1-t_value).view(-1,1,1,1)*occ_all[:,2,:].unsqueeze(1) + t_value.view(-1,1,1,1)*occ_all[:,3,:].unsqueeze(1) )
			out_l = (1-t_value).view(-1,1,1,1)*occ_all[:,0,:].unsqueeze(1)*warped_img0_l + t_value.view(-1,1,1,1)*occ_all[:,1,:].unsqueeze(1)*warped_img1_l
			if(self.args.interpOrigForw):
				out_l += (1-t_value).view(-1,1,1,1)*occ_all[:,2,:].unsqueeze(1)*x_l[:,:,0,:,:]  + t_value.view(-1,1,1,1)*occ_all[:,3,:].unsqueeze(1)*x_l[:,:,1,:,:]
			else:
				out_l += (1-t_value).view(-1,1,1,1)*occ_all[:,2,:].unsqueeze(1)*im0_tot + t_value.view(-1,1,1,1)*occ_all[:,3,:].unsqueeze(1)*im1_tot
				if(self.args.sminterpInpIm):
					out_l += (1-t_value).view(-1,1,1,1)*occ_all[:,4,:].unsqueeze(1)*x_l[:,:,0,:,:] + t_value.view(-1,1,1,1)*occ_all[:,5,:].unsqueeze(1)*x_l[:,:,1,:,:]
					divisor += (1-t_value).view(-1,1,1,1)*occ_all[:,4,:].unsqueeze(1) + t_value.view(-1,1,1,1)*occ_all[:,5,:].unsqueeze(1)
			
			out_l /=divisor
			if(not self.args.noResidAddup):
				ref_start = 6 if(self.args.sminterpInpIm) else (4 if(self.args.sminterp)else 1)
				out_l += refine_out[:, ref_start:ref_start+3, :, :]
				if(self.args.teacherflowresid and self.args.phase=="train" and level==0 and not validation):
					torch.cuda.empty_cache()
					error_res_teach = self.refine_unet_teach(torch.cat([out_l,refine_out[:, ref_start:ref_start+3, :, :],x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l,frameT],dim=1).float())
					out_l += error_res_teach
					error_res_teach = error_res_teach.detach()
					
				
		else:
			out_l = (1-t_value).view(-1,1,1,1)*occ_0_l*warped_img0_l + t_value.view(-1,1,1,1)*occ_1_l*warped_img1_l
			out_l = out_l / ( (1-t_value).view(-1,1,1,1)*occ_0_l + t_value.view(-1,1,1,1)*occ_1_l ) 
			if(not self.args.noResidAddup):
				out_l += refine_out[:, 1:4, :, :]

		


		############################		PRINTING STUFF 		###############################################
		import os,sys
		if(level == 0 and False):
			#added_pic = (added_pic*2)-1
			for i in range(8):
				#cv2.imwrite(os.path.join("tempTest", "common"+"_"+str(i)+".png"),((temp_rest[i,:].detach().to("cpu").permute(1,2,0).numpy())*255).astype(np.uint8))
				cv2.imwrite(os.path.join("tempTest", "occ_0"+"_"+str(i)+".png"),((occ_0_l[i,:].detach().to("cpu").permute(1,2,0).numpy())*255).astype(np.uint8))
				#cv2.imwrite(os.path.join("tempTest", "added_pic"+"_"+str(i)+".png"),((added_pic[i,:,:,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
				cv2.imwrite(os.path.join("tempTest", "out"+"_"+str(i)+".png"),((out_l[i,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
				#cv2.imwrite(os.path.join("tempTest", "out_ltem"+"_"+str(i)+".png"),((out_ltem[i,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
				cv2.imwrite(os.path.join("tempTest", "x_l[:,:,1,:,:]"+"_"+str(i)+".png"),((x_l[i,:,1,:,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
				cv2.imwrite(os.path.join("tempTest", "x_l[:,:,0,:,:]"+"_"+str(i)+".png"),((x_l[i,:,0,:,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
				cv2.imwrite(os.path.join("tempTest", "warped_img1_l"+"_"+str(i)+".png"),((warped_img1_l[i,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
				cv2.imwrite(os.path.join("tempTest", "warped_img0_l"+"_"+str(i)+".png"),((warped_img0_l[i,:].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
			sys.exit()
		if(self.args.temptestimages and level==0):
			multem = len(os.listdir("tempTest"))
			#warped_img1_l[warped_img1_l==0] = -1
			#warped_img0_l[warped_img0_l==0] = -1
			# Xiph: :2160,:4096
			cv2.imwrite(os.path.join("tempTest", "out"+"_"+str(multem//1)+".png"),((refine_out[0,1:4,:2160,:4096].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
			# cv2.imwrite(os.path.join("tempTest", "added_pic"+"_"+str(multem//3)+".png"),((added_pic[0,:,:2160,:4096].detach().to("cpu").permute(1,2,0).numpy())*(255)).astype(np.uint8))
			# cv2.imwrite(os.path.join("tempTest", "warped_img1_l"+"_"+str(multem//3)+".png"),((warped_img1_l[0,:,:2160,:4096].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
			# cv2.imwrite(os.path.join("tempTest", "warped_img0_l"+"_"+str(multem//3)+".png"),((warped_img0_l[0,:,:2160,:4096].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
			# cv2.imwrite(os.path.join("tempTest", "out"+"_"+str(multem//3)+".png"),((out_l[0,:,:2160,:4096].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
			# cv2.imwrite(os.path.join("tempTest", "out_pre"+"_"+str(multem//3)+".png"),((out_l_pre[0,:,:2160,:4096].detach().to("cpu").permute(1,2,0).numpy()+1)*(255/2)).astype(np.uint8))
			#sys.exit()
		#print(refine_out.shape)
		#torch_prints(refine_out[:, 0:1, :, :],"Refineout")
		

		#refine_out = 0
		
		#print("out_l shape: ",out_l.shape)



		#torch.cat()
		if(self.args.timetest):
			print("Final image generation: ",time.time()-start_time)
		if not is_training and level==0: 
			return out_l, flow_refine_l[:, 0:4, :, :] if(self.args.testgetflowout)else None#,flow_l_tmp[:,:2,:,:]

		if((self.args.additivePWC or self.args.addPWCOneFlow) and level!=0):
			flow_l -= flow_pwc
		if is_training: 
			if flow_l_prev is None and level != 0:
			# if level == self.args.S_trn:
				return out_l, flow_l,[], flow_refine_l[:, 0:4, :, :], [flow_t0_l,flow_t1_l]
			elif level != 0:
				return out_l, flow_l,[],flow_refine_l[:, 0:4, :, :], [flow_t0_l,flow_t1_l]
			else: # level==0
				return out_l, flow_l, flow_refine_l[:, 0:4, :, :], occ_0_l, [flow_t0_l,flow_t1_l], [error_res_teach,teach_flow_res]

	def get_gaussian_kernel(self,kernel_size=3, sigma=2, channels=3):
	    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
	    x_coord = torch.arange(kernel_size)
	    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
	    y_grid = x_grid.t()
	    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

	    mean = (kernel_size - 1)/2.
	    variance = sigma**2.

	    # Calculate the 2-dimensional gaussian kernel which is
	    # the product of two gaussian distributions for two different
	    # variables (in this case called x and y)
	    gaussian_kernel = (1./(2.*math.pi*variance)) *\
	                      torch.exp(
	                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
	                          (2*variance)
	                      )

	    # Make sure sum of values in gaussian kernel equals 1.
	    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

	    # Reshape to 2d depthwise convolutional weight
	    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
	    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

	    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
	                                kernel_size=kernel_size, groups=channels, bias=False,padding=1)# zero padding!

	    gaussian_filter.weight.data = gaussian_kernel
	    gaussian_filter.weight.requires_grad = False
	    
	    return gaussian_filter
	def bwarp(self, x, flo,withmask=True,minus=False):
		'''
		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow
		'''
		x_loc = x 
		if(minus or self.args.minus1bwarp):
			x_loc = (x+1)/2 

		B, C, H, W = x_loc.size()
		# mesh grid
		xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
		yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
		grid = torch.cat((xx, yy), 1).float()

		if x_loc.is_cuda:
			grid = grid.to(self.device)
		vgrid = torch.autograd.Variable(grid) + flo

		# scale grid to [-1,1]
		vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
		vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

		vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
		output = nn.functional.grid_sample(x_loc, vgrid )#align_corners=self.args.align_corners)
		mask = torch.autograd.Variable(torch.ones(x_loc.size())).to(self.device)
		mask = nn.functional.grid_sample(mask, vgrid )#align_corners=self.args.align_corners)

		# mask[mask<0.9999] = 0
		# mask[mask>0] = 1
		mask = mask.masked_fill_(mask < 0.999, 0)
		mask = mask.masked_fill_(mask > 0, 1)

		if(minus or self.args.minus1bwarp):
			output = (output*2)-1

		if(withmask):
			return output * mask
		else:
			return output

	def fwarp(self, img, flo):

		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy
			https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
		"""

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)

		N, C, _, _ = img.size()

		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1:, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)

		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get gaussian weights
		w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

		# secondly, sample each weighted corner
		img11, o11 = self.sample_one(img, x1, y1, w11)
		img12, o12 = self.sample_one(img, x1, y2, w12)
		img21, o21 = self.sample_one(img, x2, y1, w21)
		img22, o22 = self.sample_one(img, x2, y2, w22)

		imgw = img11 + img12 + img21 + img22
		o = o11 + o12 + o21 + o22

		return imgw, o


	def z_fwarp(self, img, flo, z):
		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy
			modified from https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
		"""

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)

		N, C, _, _ = img.size()

		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1:, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)

		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get gaussian weights
		w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2, z+1e-5)

		# secondly, sample each weighted corner
		img11, o11 = self.sample_one(img, x1, y1, w11)
		img12, o12 = self.sample_one(img, x1, y2, w12)
		img21, o21 = self.sample_one(img, x2, y1, w21)
		img22, o22 = self.sample_one(img, x2, y2, w22)

		imgw = img11 + img12 + img21 + img22
		o = o11 + o12 + o21 + o22

		return imgw, o


	def get_gaussian_weights(self, x, y, x1, x2, y1, y2, z=1.0):
		# z 0.0 ~ 1.0
		w11 = z * torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
		w12 = z * torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
		w21 = z * torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
		w22 = z * torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

		return w11, w12, w21, w22

	def sample_one(self, img, shiftx, shifty, weight):
		"""
		Input:
			-img (N, C, H, W)
			-shiftx, shifty (N, c, H, W)
		"""

		N, C, H, W = img.size()

		# flatten all (all restored as Tensors)
		flat_shiftx = shiftx.view(-1)
		flat_shifty = shifty.view(-1)
		flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(self.device).long().repeat(N, C,1,W).view(-1)
		flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(self.device).long().repeat(N, C,H,1).view(-1)
		flat_weight = weight.view(-1)
		flat_img = img.contiguous().view(-1)

		# The corresponding positions in I1
		idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(self.device).long().repeat(1, C, H, W).view(-1)
		idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(self.device).long().repeat(N, 1, H, W).view(-1)
		idxx = flat_shiftx.long() + flat_basex
		idxy = flat_shifty.long() + flat_basey

		# recording the inside part the shifted
		mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

		# Mask off points out of boundaries
		ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
		ids_mask = torch.masked_select(ids, mask).clone().to(self.device)

		# Note here! accmulate fla must be true for proper bp
		img_warp = torch.zeros([N * C * H * W, ]).to(self.device)
		img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

		one_warp = torch.zeros([N * C * H * W, ]).to(self.device)
		one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

		return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)


class PCARefineUNet(nn.Module):
	def __init__(self, args,teach=False):
		super(PCARefineUNet, self).__init__()
		self.args = args
		self.nf = args.nf if(not args.supLight) else 32
		self.conv1 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.conv2 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.lrelu = nn.ReLU()
		self.NN = nn.UpsamplingNearest2d(scale_factor=2)

		shuffle_scale = 4#(self.args.scales[0]/args.scales[-1])
		pixShuffAdd = 24#int(2*self.args.img_ch*self.args.dctvfi_nf/shuffle_scale)
		# INPUT MAPS
		self.input_maps = 28
		nfmul = 1
		if(not self.args.feat_input):
			self.input_maps = 16 
		if(self.args.bothflowforsynth or self.args.flowfromtto0 or self.args.bigallcomb or self.args.sminterp):
			self.input_maps = 26
		if(self.args.flowfromtto0_lite):
			self.input_maps = 20
		if(self.args.interpOrigForw):
			self.input_maps = 16
		# OUTPUT MAPS:
		self.output_maps = 1+args.img_ch
		if(self.args.sminterp):
			self.output_maps = 3 + 4
		if(self.args.sminterpInpIm):
			self.output_maps += 2
		if(self.args.noResidAddup):
			self.output_maps -= 3
			self.nf = 16
		if(self.args.smallenrefine):
			self.nf = 8
		if(self.args.lightrefine):
			self.nf = 4
			nfmul = 0.25
		if(teach):
			self.input_maps = 21
			self.output_maps = 3
			self.nf = 16
		if(args.supLight):
			self.enc0 = nn.Conv2d(16, 16, [4, 4], 2, [1, 1])
			self.dec4 = nn.Conv2d(4, 4, [3, 3], 1, [1, 1])
		self.enc1 = nn.Conv2d(self.input_maps, self.nf, [4, 4], 2, [1, 1])
		self.enc2 = nn.Conv2d(self.nf+(pixShuffAdd if(self.args.pixelshuffle)else 0), int(2*self.nf * nfmul), [4, 4], 2, [1, 1])
		self.enc3 = nn.Conv2d(int(2*self.nf * nfmul), int(4*self.nf * nfmul), [4, 4], 2, [1, 1])
		self.dec0 = nn.Conv2d(int(4*self.nf * nfmul), int(4*self.nf * nfmul), [3, 3], 1, [1, 1])
		self.dec1 = nn.Conv2d(int(4*self.nf * nfmul) + int(2*self.nf * nfmul), int(2*self.nf * nfmul), [3, 3], 1, [1, 1]) ## input concatenated with enc2
		self.dec2 = nn.Conv2d(int(2*self.nf * nfmul) + self.nf+(pixShuffAdd if(self.args.pixelshuffle)else 0), self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc1
		self.dec3 = nn.Conv2d(self.nf, self.output_maps, [3, 3], 1, [1, 1]) ## input added with warped image

	def forward(self, concat,feat_dim=0):
		#print("NF: ",self.nf)
		inp_shape = concat.shape
		if(self.args.supLight and inp_shape[2]>24):
			enc1 = self.lrelu(self.enc0(concat))
		else:
			enc1 = concat
		enc1 = self.lrelu(self.enc1(enc1))
		#print("enc1: ",enc1.shape,"featdim: ",feat_dim.shape)
		if(self.args.pixelshuffle):
			enc1 = torch.cat((enc1,feat_dim),dim=1)
		concat = 0
		torch.cuda.empty_cache()
		enc2 = self.lrelu(self.enc2(enc1))
		out = self.lrelu(self.enc3(enc2))

		out = self.lrelu(self.dec0(out))
		out = self.NN(out) # 1
		out = torch.cat((out,enc2),dim=1)
		out = self.lrelu(self.dec1(out))

		enc2 = 0
		torch.cuda.empty_cache()
		out = self.NN(out)# 2
		out = torch.cat((out,enc1),dim=1)
		out = self.lrelu(self.dec2(out))

		out = self.NN(out) # 3
		out = self.dec3(out)
		if(self.args.supLight and inp_shape[2]>24):
			out = self.NN(out)
			out = self.dec4(out)
		return out













############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################
############################################################################################################################################################################################################################







class XVFInet(nn.Module):
	
	def __init__(self, args):
		super(XVFInet, self).__init__()
		self.args = args
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
		self.nf = args.nf
		self.scale = args.module_scale_factor
		self.vfinet = VFInet(args)
		self.lrelu = nn.ReLU()
		self.in_channels = args.img_ch
		self.channel_converter = nn.Sequential(
			nn.Conv3d(self.in_channels, self.nf, [1, 3, 3], [1, 1, 1], [0, 1, 1]),	# kernelsize,stride,padding
			nn.ReLU())

		self.rec_ext_ds_module = [self.channel_converter]			# to 64 channels


		self.rfb = args.rfb
		self.extra_feature = args.extra_feat
		self.no_ds = args.no_ds_rfb
		self.part_rfb = args.no_ds_part_rfb
		if(not self.rfb):
			self.rec_ext_ds = nn.Conv3d(self.nf, self.nf, [1, 3, 3], [1, 2, 2], [0, 1, 1])
			for _ in range(int(np.log2(self.scale))):					# /4x4 also pixelanzahl durch 16
				self.rec_ext_ds_module.append(self.rec_ext_ds)
				self.rec_ext_ds_module.append(nn.ReLU())

			# This is the Feature extraction Block
			self.rec_ext_ds_module.append(nn.Conv3d(self.nf, self.nf, [1, 3, 3], 1, [0, 1, 1]))
			self.rec_ext_ds_module.append(RResBlock2D_3D(args, T_reduce_flag=False))
		else:
			self.rec_ext_ds_module = self.channel_converter
			self.myRFB = BasicRFB(self.nf,self.nf,stride=2)
			#self.feat_rfb = BasicRFB(self.nf,self.nf,stride=1)
			# here could be optional another RFB block, that does not reduce the size!! Probably not needed
			
		# Replace downscaling Conv3D with RFB-block
		if(self.no_ds):
			self.rec_ctx_ds = BasicRFB(self.nf,self.nf,stride=1,part_rfb=self.part_rfb)
		else:
			self.rec_ctx_ds = nn.Conv3d(self.nf, self.nf, [1, 3, 3], [1, 2, 2], [0, 1, 1])

		self.rec_ext_ds_module = nn.Sequential(*self.rec_ext_ds_module)

		#torch.cuda.empty_cache()

		print("The lowest scale depth for training (S_trn): ", self.args.S_trn)
		print("The lowest scale depth for test (S_tst): ", self.args.S_tst)
		print("With RFB: ", self.rfb)
		print("With extra feature RFB: ", self.extra_feature)

	def forward(self, x, t_value, is_training=True):
		'''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
		B, C, T, H, W = x.size()
		B2, C2 = t_value.size()
		assert C2 == 1, "t_value shape is [B,]"
		assert T % 2 == 0, "T must be an even number"
		t_value = t_value.view(B, 1, 1, 1)

		flow_l = None 

		torch.cuda.empty_cache()
           
		# Downscaling and Feature Extraction!!
		if(self.rfb):
			x_new = self.rec_ext_ds_module(x)
			for _ in range(int(np.log2(self.scale))):					# /4x4 also pixelanzahl durch 16
				x_new = self.myRFB(x_new)
			if(self.extra_feature):
				x_new = self.myRFB(x_new)
			feat_x = x_new
		else:
			feat_x = self.rec_ext_ds_module(x)
		# x: torch.Size([8, 3, 2, 384, 384])
		#feat_x: torch.Size([8, 64, 2, 96, 96])


		# Level Downscaling
		feat_x_list = [feat_x]
		self.lowest_depth_level = self.args.S_trn if is_training else self.args.S_tst
		if(self.part_rfb):
			eins,zwei,drei = self.rec_ctx_ds(feat_x)	
			feat_x_list.append(eins)
			feat_x_list.append(zwei)
			feat_x_list.append(drei)
		else:
			for level in range(1, self.lowest_depth_level+1):
				feat_x = self.rec_ctx_ds(feat_x)
				feat_x_list.append(feat_x)

		if is_training:
			out_l_list = []
			flow_refine_l_list = []
			out_l, flow_l, flow_refine_l = self.vfinet(x, feat_x_list[self.args.S_trn], flow_l, t_value, level=self.args.S_trn, is_training=True,no_ds=self.no_ds)
			out_l_list.append(out_l)
			flow_refine_l_list.append(flow_refine_l)
			for level in range(self.args.S_trn-1, 0, -1): ## self.args.S_trn, self.args.S_trn-1, ..., 1. level 0 is not included
				out_l, flow_l,flow_refine_l = self.vfinet(x, feat_x_list[level], flow_l, t_value, level=level, is_training=True,no_ds=self.no_ds)
				out_l_list.append(out_l)
				flow_refine_l_list.append(flow_refine_l)
			out_l, flow_l, flow_refine_l, occ_0_l0 = self.vfinet(x, feat_x_list[0], flow_l, t_value, level=0, is_training=True,no_ds=self.no_ds)
			out_l_list.append(out_l)
			flow_refine_l_list.append(flow_refine_l)
			return out_l_list[::-1], flow_refine_l_list[::-1], occ_0_l0, torch.mean(x, dim=2) # out_l_list should be reversed. [out_l0, out_l1, ...]

		else: # Testing
			for level in range(self.args.S_tst, 0, -1): ## self.args.S_tst, self.args.S_tst-1, ..., 1. level 0 is not included
				flow_l = self.vfinet(x, feat_x_list[level], flow_l, t_value, level=level, is_training=False,no_ds=self.no_ds)
			out_l = self.vfinet(x, feat_x_list[0], flow_l, t_value, level=0, is_training=False,no_ds=self.no_ds)
			return out_l







class VFInet(nn.Module):
	
	def __init__(self, args):
		super(VFInet, self).__init__()
		self.args = args
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
		self.nf = args.nf
		self.scale = args.module_scale_factor
		self.in_channels = 3

		self.conv_flow_bottom = nn.Sequential( 
			nn.Conv2d(2*self.nf, 2*self.nf, [4,4], 2, [1,1]), # kernelsize,stride,padding
			nn.ReLU(),
			nn.Conv2d(2*self.nf, 4*self.nf, [4,4], 2, [1,1]), 
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 6, [3,3], 1, [1,1]), 
			)

		self.conv_flow1 = nn.Conv2d(2*self.nf, self.nf, [3, 3], 1, [1, 1])
		
		self.conv_flow2 = nn.Sequential(
			nn.Conv2d(2*self.nf + 4, 2 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(2 * self.nf, 4 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 6, [3, 3], 1, [1, 1]),
		)

		self.conv_flow3 = nn.Sequential(
			nn.Conv2d(4 + self.nf * 4, self.nf, [1, 1], 1, [0, 0]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 2 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(2 * self.nf, 4 * self.nf, [4, 4], 2, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(4 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.UpsamplingNearest2d(scale_factor=2),
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
		)
		
		self.refine_unet = RefineUNet(args)
		self.lrelu = nn.ReLU()

	def forward(self, x, feat_x, flow_l_prev, t_value, level, is_training,no_ds=False):
		'''
		x shape : [B,C,T,H,W]
		t_value shape : [B,1] ###############
		'''
		B, C, T, H, W = x.size()
		assert T % 2 == 0, "T must be an even number"

		#print(x.shape)

		####################### For a single level 
		base = 2
		if(no_ds):
			base = 1
		l = base ** level
		x_l = x.permute(0,2,1,3,4)
		x_l = x_l.contiguous().view(B * T, C, H, W)

		if level == 0:
			pass
		else:
			x_l = F.interpolate(x_l, scale_factor=(1.0 / l, 1.0 / l), mode='bicubic', align_corners=self.args.align_cornerse)
		'''
		Down pixel-shuffle
		'''
		x_l = x_l.view(B, T, C, H//l, W//l)
		x_l = x_l.permute(0,2,1,3,4)

		B, C, T, H, W = x_l.size()

		## Feature extraction
		feat0_l = feat_x[:,:,0,:,:]
		feat1_l = feat_x[:,:,1,:,:]



# BEFORE CONV:  torch.Size([16, 64, 12, 12])
# After CONV:  torch.Size([8, 6, 12, 12])

		## Flow estimation
		if flow_l_prev is None:
			flow_l_tmp = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l), dim=1))
			flow_l = flow_l_tmp[:,:4,:,:]
			#print("Flow bottom beginning: ", flow_l.shape,feat_x.shape) # [4,12,12],[64,2,12,12] (s=3)
		else:
			up_flow_l_prev = flow_l_prev.detach() if(base ==1) else 2.0*F.interpolate(flow_l_prev.detach(), scale_factor=(2,2), mode='bilinear', align_corners=self.args.align_cornerse) 
			warped_feat1_l = self.bwarp(feat1_l, up_flow_l_prev[:,:2,:,:])
			warped_feat0_l = self.bwarp(feat0_l, up_flow_l_prev[:,2:,:,:])
			#print(up_flow_l_prev.dtype,feat0_l.dtype,warped_feat1_l.dtype)
			flow_l_tmp = self.conv_flow2(torch.cat([self.conv_flow1(torch.cat([feat0_l, warped_feat1_l],dim=1)), self.conv_flow1(torch.cat([feat1_l, warped_feat0_l],dim=1)), up_flow_l_prev],dim=1))
			flow_l = flow_l_tmp[:,:4,:,:] + up_flow_l_prev
			#print("Flow biflownet", flow_l.shape) # [4,24,24] (s=2)
		
		if not is_training and level!=0: 
			return flow_l 
		
		flow_01_l = flow_l[:,:2,:,:]
		flow_10_l = flow_l[:,2:,:,:]
		z_01_l = torch.sigmoid(flow_l_tmp[:,4:5,:,:])
		z_10_l = torch.sigmoid(flow_l_tmp[:,5:6,:,:])
		
		## Complementary Flow Reversal (CFR)
		flow_forward, norm0_l = self.z_fwarp(flow_01_l, t_value * flow_01_l, z_01_l)  ## Actually, F (t) -> (t+1). Translation only. Not normalized yet
		flow_backward, norm1_l = self.z_fwarp(flow_10_l, (1-t_value) * flow_10_l, z_10_l)  ## Actually, F (1-t) -> (-t). Translation only. Not normalized yet
		
		
		flow_t0_l = -(1-t_value) * ((t_value)*flow_forward) + (t_value) * ((t_value)*flow_backward) # The numerator of Eq.(1) in the paper.
		flow_t1_l = (1-t_value) * ((1-t_value)*flow_forward) - (t_value) * ((1-t_value)*flow_backward) # The numerator of Eq.(2) in the paper.
		
		norm_l = (1-t_value)*norm0_l + t_value*norm1_l
		mask_ = (norm_l.detach() > 0).type(norm_l.type())
		flow_t0_l = (1-mask_) * flow_t0_l + mask_ * (flow_t0_l.clone() / (norm_l.clone() + (1-mask_))) # Divide the numerator with denominator in Eq.(1)
		flow_t1_l = (1-mask_) * flow_t1_l + mask_ * (flow_t1_l.clone() / (norm_l.clone() + (1-mask_))) # Divide the numerator with denominator in Eq.(2)

		## Feature warping
		warped0_l = self.bwarp(feat0_l, flow_t0_l)
		warped1_l = self.bwarp(feat1_l, flow_t1_l)

		## Flow refinement
		flow_refine_l = torch.cat([feat0_l, warped0_l, warped1_l, feat1_l, flow_t0_l, flow_t1_l], dim=1)
		#print(flow_refine_l.dtype,flow_t0_l.dtype,flow_t1_l.dtype)
		if(self.args.halfXVFI):
			flow_refine_l = self.conv_flow3(flow_refine_l.half()) + torch.cat([flow_t0_l.half(), flow_t1_l.half()], dim=1)
		else:
			flow_refine_l = self.conv_flow3(flow_refine_l) + torch.cat([flow_t0_l, flow_t1_l], dim=1)
		#print("TFFlowNet flow: ",flow_refine_l.shape)# [4,12,12] (s=3)
		flow_t0_l = flow_refine_l[:, :2, :, :]
		flow_t1_l = flow_refine_l[:, 2:4, :, :]

		# Warped Features
		warped0_l = self.bwarp(feat0_l, flow_t0_l)
		warped1_l = self.bwarp(feat1_l, flow_t1_l)

		## Flow upscale
		flow_t0_l = self.scale * F.interpolate(flow_t0_l, scale_factor=(self.scale, self.scale), mode='bilinear',align_corners=self.args.align_cornerse)
		flow_t1_l = self.scale * F.interpolate(flow_t1_l, scale_factor=(self.scale, self.scale), mode='bilinear',align_corners=self.args.align_cornerse)

		## Image warping and blending
		warped_img0_l = self.bwarp(x_l[:,:,0,:,:], flow_t0_l)
		warped_img1_l = self.bwarp(x_l[:,:,1,:,:], flow_t1_l)
		
		if(self.args.no_refine):
			out_l = (1-t_value)*warped_img0_l + t_value*warped_img1_l
			occ_0_l = 0
		else:
			refine_out = self.refine_unet(torch.cat([F.pixel_shuffle(torch.cat([feat0_l, feat1_l, warped0_l, warped1_l],dim=1), self.scale), x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l],dim=1))
			occ_0_l = torch.sigmoid(refine_out[:, 0:1, :, :])
			occ_1_l = 1-occ_0_l
			
			out_l = (1-t_value)*occ_0_l*warped_img0_l + t_value*occ_1_l*warped_img1_l
			out_l = out_l / ( (1-t_value)*occ_0_l + t_value*occ_1_l ) + refine_out[:, 1:4, :, :]

		if not is_training and level==0: 
			return out_l

		if is_training: 
			if flow_l_prev is None:
			# if level == self.args.S_trn:
				return out_l, flow_l, flow_refine_l[:, 0:4, :, :]
			elif level != 0:
				return out_l, flow_l,flow_refine_l[:, 0:4, :, :]
			else: # level==0
				return out_l, flow_l, flow_refine_l[:, 0:4, :, :], occ_0_l

	def bwarp(self, x, flo):
		'''
		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow
		'''
		B, C, H, W = x.size()
		# mesh grid
		xx = torch.arange(0, W).view(1, 1, 1, W).expand(B, 1, H, W)
		yy = torch.arange(0, H).view(1, 1, H, 1).expand(B, 1, H, W)
		grid = torch.cat((xx, yy), 1).float()

		if x.is_cuda:
			grid = grid.to(self.device)
		vgrid = torch.autograd.Variable(grid) + flo

		# scale grid to [-1,1]
		vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
		vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

		vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
		if(self.args.halfXVFI):
			output = nn.functional.grid_sample(x, vgrid.half() )#align_corners=self.args.align_corners)
		else:
			output = nn.functional.grid_sample(x, vgrid )#align_corners=self.args.align_corners)

		mask = torch.autograd.Variable(torch.ones(x.size())).to(self.device)
		mask = nn.functional.grid_sample(mask, vgrid )#align_corners=self.args.align_corners)

		# mask[mask<0.9999] = 0
		# mask[mask>0] = 1
		mask = mask.masked_fill_(mask < 0.999, 0)
		mask = mask.masked_fill_(mask > 0, 1)

		#if(self.args.with)
		if(self.args.halfXVFI):
			return (output * mask).half()
		else:
			return (output * mask)

	def fwarp(self, img, flo):

		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy
			https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
		"""

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)

		N, C, _, _ = img.size()

		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1:, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)

		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get gaussian weights
		w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2)

		# secondly, sample each weighted corner
		img11, o11 = self.sample_one(img, x1, y1, w11)
		img12, o12 = self.sample_one(img, x1, y2, w12)
		img21, o21 = self.sample_one(img, x2, y1, w21)
		img22, o22 = self.sample_one(img, x2, y2, w22)

		imgw = img11 + img12 + img21 + img22
		o = o11 + o12 + o21 + o22

		return imgw, o


	def z_fwarp(self, img, flo, z):
		"""
			-img: image (N, C, H, W)
			-flo: optical flow (N, 2, H, W)
			elements of flo is in [0, H] and [0, W] for dx, dy
			modified from https://github.com/lyh-18/EQVI/blob/EQVI-master/models/forward_warp_gaussian.py
		"""

		# (x1, y1)		(x1, y2)
		# +---------------+
		# |				  |
		# |	o(x, y) 	  |
		# |				  |
		# |				  |
		# |				  |
		# |				  |
		# +---------------+
		# (x2, y1)		(x2, y2)

		N, C, _, _ = img.size()

		# translate start-point optical flow to end-point optical flow
		y = flo[:, 0:1:, :]
		x = flo[:, 1:2, :, :]

		x = x.repeat(1, C, 1, 1)
		y = y.repeat(1, C, 1, 1)

		# Four point of square (x1, y1), (x1, y2), (x2, y1), (y2, y2)
		x1 = torch.floor(x)
		x2 = x1 + 1
		y1 = torch.floor(y)
		y2 = y1 + 1

		# firstly, get gaussian weights
		w11, w12, w21, w22 = self.get_gaussian_weights(x, y, x1, x2, y1, y2, z+1e-5)

		# secondly, sample each weighted corner
		img11, o11 = self.sample_one(img, x1, y1, w11)
		img12, o12 = self.sample_one(img, x1, y2, w12)
		img21, o21 = self.sample_one(img, x2, y1, w21)
		img22, o22 = self.sample_one(img, x2, y2, w22)

		imgw = img11 + img12 + img21 + img22
		o = o11 + o12 + o21 + o22

		if(self.args.halfXVFI):
			return imgw.half(), o.half()
		else:
			return imgw, o


	def get_gaussian_weights(self, x, y, x1, x2, y1, y2, z=1.0):
		# z 0.0 ~ 1.0
		w11 = z * torch.exp(-((x - x1) ** 2 + (y - y1) ** 2))
		w12 = z * torch.exp(-((x - x1) ** 2 + (y - y2) ** 2))
		w21 = z * torch.exp(-((x - x2) ** 2 + (y - y1) ** 2))
		w22 = z * torch.exp(-((x - x2) ** 2 + (y - y2) ** 2))

		return w11, w12, w21, w22

	def sample_one(self, img, shiftx, shifty, weight):
		"""
		Input:
			-img (N, C, H, W)
			-shiftx, shifty (N, c, H, W)
		"""

		N, C, H, W = img.size()

		# flatten all (all restored as Tensors)
		flat_shiftx = shiftx.view(-1)
		flat_shifty = shifty.view(-1)
		flat_basex = torch.arange(0, H, requires_grad=False).view(-1, 1)[None, None].to(self.device).long().repeat(N, C,1,W).view(-1)
		flat_basey = torch.arange(0, W, requires_grad=False).view(1, -1)[None, None].to(self.device).long().repeat(N, C,H,1).view(-1)
		flat_weight = weight.view(-1)
		flat_img = img.contiguous().view(-1)

		# The corresponding positions in I1
		idxn = torch.arange(0, N, requires_grad=False).view(N, 1, 1, 1).to(self.device).long().repeat(1, C, H, W).view(-1)
		idxc = torch.arange(0, C, requires_grad=False).view(1, C, 1, 1).to(self.device).long().repeat(N, 1, H, W).view(-1)
		idxx = flat_shiftx.long() + flat_basex
		idxy = flat_shifty.long() + flat_basey

		# recording the inside part the shifted
		mask = idxx.ge(0) & idxx.lt(H) & idxy.ge(0) & idxy.lt(W)

		# Mask off points out of boundaries
		ids = (idxn * C * H * W + idxc * H * W + idxx * W + idxy)
		ids_mask = torch.masked_select(ids, mask).clone().to(self.device)

		# Note here! accmulate fla must be true for proper bp
		img_warp = torch.zeros([N * C * H * W, ]).to(self.device)
		img_warp.put_(ids_mask, torch.masked_select(flat_img * flat_weight, mask), accumulate=True)

		one_warp = torch.zeros([N * C * H * W, ]).to(self.device)
		one_warp.put_(ids_mask, torch.masked_select(flat_weight, mask), accumulate=True)

		return img_warp.view(N, C, H, W), one_warp.view(N, C, H, W)




class RefineUNet(nn.Module):
	def __init__(self, args):
		super(RefineUNet, self).__init__()
		self.args = args
		self.scale = args.module_scale_factor
		self.nf = args.nf
		self.conv1 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.conv2 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.lrelu = nn.ReLU()
		self.NN = nn.UpsamplingNearest2d(scale_factor=2)

		self.enc1 = nn.Conv2d((4*self.nf)//self.scale//self.scale + 4*args.img_ch + 4, self.nf, [4, 4], 2, [1, 1])
		self.enc2 = nn.Conv2d(self.nf, 2*self.nf, [4, 4], 2, [1, 1])
		self.enc3 = nn.Conv2d(2*self.nf, 4*self.nf, [4, 4], 2, [1, 1])
		self.dec0 = nn.Conv2d(4*self.nf, 4*self.nf, [3, 3], 1, [1, 1])
		self.dec1 = nn.Conv2d(4*self.nf + 2*self.nf, 2*self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc2
		self.dec2 = nn.Conv2d(2*self.nf + self.nf, self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc1
		self.dec3 = nn.Conv2d(self.nf, 1+args.img_ch, [3, 3], 1, [1, 1]) ## input added with warped image

	def forward(self, concat):
		enc1 = self.lrelu(self.enc1(concat))
		enc2 = self.lrelu(self.enc2(enc1))
		out = self.lrelu(self.enc3(enc2))

		out = self.lrelu(self.dec0(out))
		out = self.NN(out)

		out = torch.cat((out,enc2),dim=1)
		out = self.lrelu(self.dec1(out))

		out = self.NN(out)
		out = torch.cat((out,enc1),dim=1)
		out = self.lrelu(self.dec2(out))

		out = self.NN(out)
		out = self.dec3(out)
		return out

class ResBlock2D_3D(nn.Module):
	## Shape of input [B,C,T,H,W]
	## Shape of output [B,C,T,H,W]
	def __init__(self, args):
		super(ResBlock2D_3D, self).__init__()
		self.args = args
		self.nf = args.nf

		self.conv3x3_1 = nn.Conv3d(self.nf, self.nf, [1,3,3], 1, [0,1,1])
		self.conv3x3_2 = nn.Conv3d(self.nf, self.nf, [1,3,3], 1, [0,1,1])
		self.lrelu = nn.ReLU()

	def forward(self, x):
		'''
		x shape : [B,C,T,H,W]
		'''
		B, C, T, H, W = x.size()

		out = self.conv3x3_2(self.lrelu(self.conv3x3_1(x)))

		return x + out

class RResBlock2D_3D(nn.Module):
	
	def __init__(self, args, T_reduce_flag=False):
		super(RResBlock2D_3D, self).__init__()
		self.args = args
		self.nf = args.nf
		self.T_reduce_flag = T_reduce_flag
		self.resblock1 = ResBlock2D_3D(self.args)
		self.resblock2 = ResBlock2D_3D(self.args)
		if T_reduce_flag:
			self.reduceT_conv = nn.Conv3d(self.nf, self.nf, [3,1,1], 1, [0,0,0])

	def forward(self, x):
		'''
		x shape : [B,C,T,H,W]
		'''
		out = self.resblock1(x)
		out = self.resblock2(out)
		if self.T_reduce_flag:
			return self.reduceT_conv(out + x)
		else:
			return out + x
