# This code contains parts of XVFInet from Sim et al. (https://github.com/JihyongOh/XVFI) 
# Their extensive code and Dataset were crucial for this.

import functools, random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

import math
from skimage import feature
import cv2
import sys

from useful import torch_prints,numpy_prints,MyPWC

from pca_comp import pca_inverse,to_pca_diff

import cupy as cp
import time
from softSplat import Softsplat


class DCTXVFInet(nn.Module):
	
	def __init__(self, args):
		super(DCTXVFInet, self).__init__()
		self.args = args
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  
		self.lrelu = nn.ReLU()
		self.in_channels = args.img_ch
		
		self.output_size = (args.patch_size,args.patch_size)	
		

		self.output_size_val = (self.args.validation_patch_size,self.args.validation_patch_size)
		self.output_size_test = (2160,4096)
		self.nf = int(self.args.dctvfi_nf  )

		self.base_modules = nn.ModuleList([])
		
		if(self.args.ref_feat_extrac):
			self.rec_ctx_ds = nn.Sequential(
				nn.Conv2d(self.args.dctvfi_nf*6 , self.nf*self.args.img_ch*2, 3, 1, 1),
				nn.ReLU(),
				nn.Conv2d(self.nf*6, self.args.dctvfi_nf*self.args.img_ch*2, 3, 1, 1),
				nn.ReLU()
				)
			
			self.base_modules.append(self.rec_ctx_ds)

		
		self.vfinet = DCTVFInet(args,self.output_size,self.output_size_test,self.output_size_val)
		self.base_modules.append(self.vfinet)
		self.mypwc = None

		res = sum(p.numel() for p in self.vfinet.parameters())
		print("Parameters of DCTVFInet: ",res)
		
		if(self.args.optimizeEV):
			
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



		self.used_pcas = None
		self.params = None
		print("The lowest scale depth for training (S_trn): ", self.args.S_trn)
		print("The lowest scale depth for test (S_tst): ", self.args.S_tst)




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

		t_value = t_value.view(B, 1, 1, 1)
		x_l = normInput
		orig_images = x_l[0]

		
		flow_l = None
		needed = max(self.args.S_tst,self.args.S_trn) + 1
		for i in range(len(input_gpuList) - needed):
			input_gpuList[i+needed] = None
		torch.cuda.empty_cache()

		# PCA Conversion
		start_time = time.time()
		
		for i in range(self.args.S_tst+1):
			input_gpuList[i] = torch.zeros(input_gpuList[i].shape,device=self.args.gpu).float()
			tempMul = 8/self.args.scales[i]
			index8 = self.args.scales.index(8)

			
			for temIndex,temEvs in enumerate(self.EVs):
				if(temEvs == None):
					continue
				assert not temEvs.isnan().any(), "Any EV "+str(temIndex)
			
			if(self.args.noEVOptimization):
				assert not self.pca_means[index8].requires_grad and not self.EVs[index8].requires_grad, "requires grad is true!!!"
			input_gpuList[i] = to_pca_diff(x_l[i].reshape(B*6,int(H2*tempMul),int(W2*tempMul)),self.params[i],self.args,self.pca_means[index8],self.EVs[index8],self.mean_vecs[index8]).reshape(B,self.args.dctvfi_nf*6,int(H2*tempMul//8),int(W2*tempMul//8)).float()
				
		tempInp = 0
		torch.cuda.empty_cache()

		if(self.args.meanVecParam):
			for i in self.mean_vecs:
				if(i == None):
					continue
				assert not i.requires_grad 
		
		start_time = time.time()

		feat_x_list = [] 
		for i in range(needed):
			if(self.args.ref_feat_extrac):
				feat_x_list.append(self.rec_ctx_ds(input_gpuList[i])+input_gpuList[i])
					

		input_gpuList = 0 
		torch.cuda.empty_cache()
		pwcflow_list = []
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
			
			for level in range(self.args.S_trn-1, 0, -1):
				out_l, flow_l, maskList, flow_refine_l, endflow = self.vfinet(feat_x_list[level], flow_l, t_value, level=level, is_training=True,normInput=x_l[level],validation=validation,epoch=epoch,feat_pyr=pwcflow_list[level],orig_images=orig_images)
				out_l_list.append(out_l)
				flow_refine_l_list.append(flow_refine_l)
				unrefined_flow_list.append(flow_l)
				endflow_list.append(endflow)
				torch.cuda.empty_cache()

			out_l, flow_l, flow_refine_l, occ_0_l0, endflow = self.vfinet(feat_x_list[0], flow_l, t_value, level=0, is_training=True,normInput=x_l[0],validation=validation,epoch=epoch,feat_pyr=pwcflow_list[0],mypwc=None,orig_images=orig_images,frameT=frameT)
			out_l_list.append(out_l)
			flow_refine_l_list.append(flow_refine_l)
			unrefined_flow_list.append(flow_l)
			endflow_list.append(endflow)
			

			flow_refine_l_list = flow_refine_l_list[::-1]
			out_l_list = out_l_list[::-1]
			unrefined_flow_list =unrefined_flow_list[::-1]
			endflow_list = endflow_list[::-1]

			
			mean_pics = torch.mean(orig_images,dim=2)
			
			return out_l_list, flow_refine_l_list,unrefined_flow_list, occ_0_l0, mean_pics,endflow_list 

		else: # Testing
			for level in range(self.args.S_tst, 0, -1): 
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
				if(self.args.optimizeEV):  
					index = 0
					
					with torch.no_grad():
						for i in self.used_pcas:
							if(self.args.phase=="train"):
								temp = torch.as_tensor(cp.asnumpy(i.mean),device=self.args.gpu)
								temp.requires_grad= not self.args.noEVOptimization
								
								if(not self.args.ExacOneEV or index==0):
									self.pca_means[index][:] = temp 
									self.pca_means[index].requires_grad = not self.args.noEVOptimization
								temp = torch.as_tensor(cp.asnumpy(i.eigenvectors),device=self.args.gpu)
								temp.requires_grad= not self.args.noEVOptimization
								if(not self.args.ExacOneEV or index==0):
									
									self.EVs[index][:] = temp
									self.EVs[index].requires_grad = not self.args.noEVOptimization
								
							if(not self.args.meanVecParam):
								self.mean_vecs[index] = torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)
							else:
								if(not self.args.ExacOneEV or index==0):
									self.mean_vecs[index][:] = torch.as_tensor(i.store["mean_vec"],device=self.args.gpu)

							index += 1
		self.used_pcas = None
	def pick_norm_vec(self,pca):
		if(self.used_pcas == None):
			self.used_pcas = pca
			for i in self.used_pcas:
				i.store["mima"] = (i.store["mima"][0].to(self.args.gpu),i.store["mima"][1].to(self.args.gpu))
			if(self.args.optimizeEV): 
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
		self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  
		self.nf = int(self.args.dctvfi_nf * self.args.img_ch ) 
		self.in_channels = 3
		self.output_size = output_size
		self.output_size_test = output_size_test
		self.output_size_val = output_size_val		
		


		self.softsplat = Softsplat()
		
		self.conv_flow_bottom = nn.Sequential( 
			nn.Conv2d(2*self.args.dctvfi_nf * self.args.img_ch , 2*self.nf, [3,3], 1, [1,1]), 
			nn.ReLU(),
			nn.Conv2d(2*self.nf, 2*self.nf, [3,3], 1, [1,1]), 
			nn.ReLU(),
			
			nn.Conv2d(2 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			
			nn.Conv2d(2 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf,4 if(self.args.cutoffUnnec and not self.args.tempbottomflowfix)else 6, [3,3], 1, [1,1]), 
			)

		
		self.conv_flow1 = nn.Conv2d(2 *self.args.dctvfi_nf * self.args.img_ch, self.nf, [3, 3], 1, [1, 1])

		self.conv_flow2 = nn.Sequential(
			nn.Conv2d(2*self.nf + 4, 2 * self.nf, [3, 3], 1, [1,1]), 
			nn.ReLU(),
			nn.Conv2d(2 * self.nf, 2 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(2 * self.nf, 1 * self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(1 * self.nf, self.nf, [3, 3], 1, [1, 1]),
			nn.ReLU(),
			nn.Conv2d(self.nf, 4, [3, 3], 1, [1, 1]),
		)
		
		


		

		self.refine_unet = PCARefineUNet(args)
		res = sum(p.numel() for p in self.refine_unet.parameters())
		print("Parameters of refine UNET: ",res)
		self.lrelu = nn.ReLU()
		if(self.args.sminterp):
			self.T_param =  torch.nn.Parameter(torch.ones((1),device=self.args.gpu).double())
			self.T_param.requires_grad = False
		if(self.args.impmasksoftsplat):
			self.z_alpha =  torch.nn.Parameter(torch.ones((2),device=self.args.gpu).double())
		


	def forward(self, feat_x, flow_l_prev, t_value, level, is_training,normInput=0,validation=False,epoch=0,feat_pyr=[],mypwc=[],orig_images=None,frameT=None):
		B,C,H,W = feat_x.shape
		

		feat_x = feat_x.reshape(B,2,self.args.img_ch*self.args.dctvfi_nf,H,W)
		feat0_l = feat_x[:,0,:,:,:]
		feat1_l = feat_x[:,1,:,:,:]
		torch.cuda.empty_cache()
		x_l = normInput




		start_time = time.time()
		## Flow estimation
		if flow_l_prev is None:
			
			flow_l_tmp = self.conv_flow_bottom(torch.cat((feat0_l, feat1_l), dim=1))
			flow_l = flow_l_tmp[:,:4,:,:]
		else:

			temp_shap = flow_l_prev.shape 
			up_flow_l_prev = F.interpolate(flow_l_prev.detach(), size=(feat1_l.shape[2],feat1_l.shape[3]), mode='bilinear', align_corners=self.args.align_cornerse)
			up_flow_l_prev *= (up_flow_l_prev.shape[3]/temp_shap[3])
			warped_feat1_l = self.softsplat(feat1_l, up_flow_l_prev[:,:2,:,:])
			warped_feat0_l = self.softsplat(feat0_l, up_flow_l_prev[:,2:,:,:])
		
			flow_l_tmp = self.conv_flow2(torch.cat([self.conv_flow1(torch.cat([feat0_l, warped_feat1_l],dim=1)), self.conv_flow1(torch.cat([feat1_l, warped_feat0_l],dim=1)), up_flow_l_prev],dim=1))
			
			flow_l = flow_l_tmp[:,:4,:,:] + up_flow_l_prev 




		if not is_training and level!=0: 
			return flow_l 
		
		
		flow_10_l = flow_l[:,:2,:,:]
		flow_01_l = flow_l[:,2:,:,:]
		

		flow_t0_l = t_value.view(-1,1,1,1) * flow_01_l
		flow_t1_l = (1 - t_value).view(-1,1,1,1) * flow_10_l
		if(self.args.phase != "test" or self.args.testgetflowout):
			flow_refine_l =  torch.cat([flow_t0_l, flow_t1_l], dim=1)

	


		upscaleDict = [self.args.scales[0]/1.0 for i in self.args.scales] 
		upscale =   x_l.shape[3]/flow_t0_l.shape[2]  

		if(not upscale.is_integer()):
			raise Exception("upscale factor is no integer!!! Upscale factor: " + str(upscale))
		upscale = int(upscale)
		
		if(upscale < 1.1 and upscale >0.9):
			raise Exception("Well there should be some upsampling here.")
		else:
			flow_t0_l =  upscale * F.interpolate(flow_t0_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
			flow_t1_l = upscale * F.interpolate(flow_t1_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
			flow_10_l = upscale * F.interpolate(flow_10_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
			flow_01_l = upscale * F.interpolate(flow_01_l, scale_factor=(upscale, upscale), mode='bilinear',align_corners=self.args.align_cornerse)
			
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
		flowback_0 = self.bwarp(flow_10_l*t_value, (1 -t_value)*flow_01_l,withmask=not self.args.outMaskLess) # t->0
		flowback_1 = self.bwarp(flow_01_l*(1 -t_value), t_value*flow_10_l,withmask=not self.args.outMaskLess) # t -> 1

		torch.cuda.empty_cache()
		im0_tot =  self.bwarp(x_l[:,:,0,:,:], flowback_0,withmask=not self.args.outMaskLess)
		im1_tot =  self.bwarp(x_l[:,:,1,:,:], flowback_1,withmask=not self.args.outMaskLess)
		refine_out = torch.cat([ x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l,flowback_0,flowback_1,im0_tot,im1_tot],dim=1)
		

		flowback_1 = 0
		flowback_0 = 0
		
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

		refine_out = self.refine_unet(refine_out)
		torch.cuda.empty_cache()

		feat0_l =0
		feat1_l =0
		warped0_l = 0
		warped1_l = 0

		assert self.args.TOptimization == self.T_param.requires_grad
		num_softmax_combs = 6	
		occ_all = F.softmax(refine_out[:, 0:num_softmax_combs, :, :]/self.T_param,dim=1)
		occ_0_l = occ_all[:,0:1,:,:]
	
		

		################################# 		IMAGE SYNTHESIS 			##########################################
		divisor = ( (1-t_value).view(-1,1,1,1)*occ_all[:,0,:].unsqueeze(1) + t_value.view(-1,1,1,1)*occ_all[:,1,:].unsqueeze(1) + (1-t_value).view(-1,1,1,1)*occ_all[:,2,:].unsqueeze(1) + t_value.view(-1,1,1,1)*occ_all[:,3,:].unsqueeze(1) )
		out_l = (1-t_value).view(-1,1,1,1)*occ_all[:,0,:].unsqueeze(1)*warped_img0_l + t_value.view(-1,1,1,1)*occ_all[:,1,:].unsqueeze(1)*warped_img1_l
	
		out_l += (1-t_value).view(-1,1,1,1)*occ_all[:,2,:].unsqueeze(1)*im0_tot + t_value.view(-1,1,1,1)*occ_all[:,3,:].unsqueeze(1)*im1_tot
		out_l += (1-t_value).view(-1,1,1,1)*occ_all[:,4,:].unsqueeze(1)*x_l[:,:,0,:,:] + t_value.view(-1,1,1,1)*occ_all[:,5,:].unsqueeze(1)*x_l[:,:,1,:,:]
		divisor += (1-t_value).view(-1,1,1,1)*occ_all[:,4,:].unsqueeze(1) + t_value.view(-1,1,1,1)*occ_all[:,5,:].unsqueeze(1)
		
		out_l /=divisor
		

		


		############################		PRINTING STUFF 		###############################################
		import os,sys
		

		if not is_training and level==0: 
			return out_l, flow_refine_l[:, 0:4, :, :] if(self.args.testgetflowout)else None

		if is_training: 
			if flow_l_prev is None and level != 0:
				return out_l, flow_l,[], flow_refine_l[:, 0:4, :, :], [flow_t0_l,flow_t1_l]
			elif level != 0:
				return out_l, flow_l,[],flow_refine_l[:, 0:4, :, :], [flow_t0_l,flow_t1_l]
			else: # level==0
				return out_l, flow_l, flow_refine_l[:, 0:4, :, :], occ_0_l, [flow_t0_l,flow_t1_l]

	
	def bwarp(self, x, flo,withmask=True,minus=False):
		'''
		x: [B, C, H, W] (im2)
		flo: [B, 2, H, W] flow
		'''
		x_loc = x 
		
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
		output = nn.functional.grid_sample(x_loc, vgrid )
		mask = torch.autograd.Variable(torch.ones(x_loc.size())).to(self.device)
		mask = nn.functional.grid_sample(mask, vgrid )

		
		mask = mask.masked_fill_(mask < 0.999, 0)
		mask = mask.masked_fill_(mask > 0, 1)

		

		if(withmask):
			return output * mask
		else:
			return output

	
class PCARefineUNet(nn.Module):
	def __init__(self, args,teach=False):
		super(PCARefineUNet, self).__init__()
		self.args = args
		self.nf = args.nf 
		self.conv1 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.conv2 = nn.Conv2d(self.nf, self.nf, [3,3], 1, [1,1])
		self.lrelu = nn.ReLU()
		self.NN = nn.UpsamplingNearest2d(scale_factor=2)

		shuffle_scale = 4
		# INPUT MAPS
		self.input_maps = 28
		nfmul = 1
		if(self.args.sminterp):
			self.input_maps = 26
		
		# OUTPUT MAPS:
		self.output_maps = 1+args.img_ch
		if(self.args.sminterp):
			self.output_maps = 3 + 4
		if(self.args.sminterpInpIm):
			self.output_maps += 2
		if(self.args.noResidAddup):
			self.output_maps -= 3
			self.nf = 16
		
		self.enc1 = nn.Conv2d(self.input_maps, self.nf, [4, 4], 2, [1, 1])
		self.enc2 = nn.Conv2d(self.nf, int(2*self.nf * nfmul), [4, 4], 2, [1, 1])
		self.enc3 = nn.Conv2d(int(2*self.nf * nfmul), int(4*self.nf * nfmul), [4, 4], 2, [1, 1])
		self.dec0 = nn.Conv2d(int(4*self.nf * nfmul), int(4*self.nf * nfmul), [3, 3], 1, [1, 1])
		self.dec1 = nn.Conv2d(int(4*self.nf * nfmul) + int(2*self.nf * nfmul), int(2*self.nf * nfmul), [3, 3], 1, [1, 1]) ## input concatenated with enc2
		self.dec2 = nn.Conv2d(int(2*self.nf * nfmul) + self.nf, self.nf, [3, 3], 1, [1, 1]) ## input concatenated with enc1
		self.dec3 = nn.Conv2d(self.nf, self.output_maps, [3, 3], 1, [1, 1]) ## input added with warped image

	def forward(self, concat,feat_dim=0):
		
		inp_shape = concat.shape
		enc1 = concat
		enc1 = self.lrelu(self.enc1(enc1))
		

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
		return out










