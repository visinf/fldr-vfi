from dataclasses import dataclass
import torch.utils.data as data
import torch.nn as nn
import os, glob, sys, torch, shutil, random, math, time, cv2
import torch
import numpy as np
import sys
import math

import pickle as pk
import time
from functools import reduce  # Required in Python 3
import operator
from skimage.metrics import peak_signal_noise_ratio

#import pycuda.autoinit
#import pycuda.gpuarray as gpuarray
#import skcuda.linalg as linalg
#from skcuda.linalg import PCA as cuPCA

from sklearn.decomposition import PCA
from numpy.lib.stride_tricks import as_strided
import scipy.fft as scF
import cupy as cp
#import cupyx.scipy.fft as cufft
from myAdditions import MYPCA,torch_prints,numpy_prints


# import numpy as cp
# import scipy.fft as cufft
# from sklearn.decomposition import PCA as MYPCA
#from cuml.decomposition import PCA
#import cuml
# My files import:



def my_RGBframes_np2Tensor(imgIn, channel,on_gpu):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (channel, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    if on_gpu:
        imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)
    else:
        imgIn = imgIn.transpose(ts).astype(float)#.mul_(1.0)
    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn

def prod(a):
    return reduce(operator.mul,a,1)


def my_frames_loader_train(params, candidate_frames, frameRange):
    frames = []
    for frameIndex in frameRange:
        frame = cv2.imread(candidate_frames[frameIndex])
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if params.need_patch:  ## random crop
        ps = params.patch_size
        ix = random.randrange(0, iw - ps + 1)
        iy = random.randrange(0, ih - ps + 1)
        frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    if random.random() < 0.5:  # random horizontal flip
        frames = frames[:, :, ::-1, :]

    # No vertical flip

    rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ np2Tensor [-1,1] normalized """
    frames = my_RGBframes_np2Tensor(frames, params.img_ch,params.on_gpu)

    return frames

def my_make_2D_dataset_X_Train(dir):
    framesPath = []
    # Find and loop over all the clips in root `dir`.
    for scene_path in sorted(glob.glob(os.path.join(dir, '*', ''))):
        sample_paths = sorted(glob.glob(os.path.join(scene_path, '*', '')))
        for sample_path in sample_paths:
            frame65_list = []
            for frame in sorted(glob.glob(os.path.join(sample_path, '*.png'))):
                frame65_list.append(frame)
            framesPath.append(frame65_list)

    print("The number of total training samples : {} which has 65 frames each.".format(
        len(framesPath)))  ## 4408 folders which have 65 frames each
    return framesPath

class X_Train(data.Dataset):
    def __init__(self,params):
        self.params = params
        self.max_t_step_size = params.max_t_step_size
        self.framesPath = my_make_2D_dataset_X_Train(self.params.train_path)
        self.nScenes = len(self.framesPath)

        # Raise error if no images found in train_data_path.
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.params.train_path + "\n"))

    def __getitem__(self, idx):
        t_step_size = random.randint(2, self.max_t_step_size)
        t_list = np.linspace((1 / t_step_size), (1 - (1 / t_step_size)), (t_step_size - 1))

        candidate_frames = self.framesPath[idx]
        firstFrameIdx = random.randint(0, (64 - t_step_size))
        interIdx = random.randint(1, t_step_size - 1)  # relative index, 1~self.t_step_size-1
        interFrameIdx = firstFrameIdx + interIdx  # absolute index
        t_value = t_list[interIdx - 1]  # [0,1]

        ########    TEMPORAL DATA AUGMENTATION      ####################
        if (random.randint(0, 1)):
            frameRange = [firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + t_step_size, firstFrameIdx, interFrameIdx]
            interIdx = t_step_size - interIdx  # (self.t_step_size-1) ~ 1
            t_value = 1.0 - t_value

        ##########      SPATIAL DATA AUGMENTATION       ################################
        frames = my_frames_loader_train(self.params, candidate_frames,
                                     frameRange)  # including "np2Tensor [-1,1] normalized"

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes

def clipping(a):
    a =  np.where(a<0 ,0,a  )
    return np.where(a>1 ,1,a  )




def rgb2gray(im):
    return  0.2125 *im[0,:] + 0.7154 *im[1,:] + 0.0721 *im[2,:]

def create_pca(dataloader,params):
    images = []
    wiS = params.wiS
    stored_eighter = np.zeros((params.nSamples,wiS*wiS))
    se_index = 0
    for trainIndex, (frames, t_value) in enumerate(dataloader):
        input_frames = frames[:, :, :-1, :] # [B, C, T, H, W]
        frameT = frames[:, :, -1, :]  # [B, C, H, W]
        sized = (input_frames.shape[3]//wiS)*wiS
        
        input_frames[0,:,:,:,:] =(input_frames[0,:,:,:,:] +1)/2
        frameT[0,:,:,:] = (frameT[0,:,:,:]+1)/2

        print(input_frames.shape)
        print("max, min: ",torch.max(input_frames),torch.min(input_frames))

        images = [rgb2gray(input_frames[0,:,0,:sized,:sized]),rgb2gray(input_frames[0,:,1,:sized,:sized]),rgb2gray(frameT[0,:,:sized,:sized])] #[3,sized,sized]

        # 388,584 Padding to 8 dividable
        # padded_images = []
        # padding_h = wiS - images[0].shape[0]%wiS 
        # padding_w = wiS - images[0].shape[1]%wiS
        # for i in images:
        #     temp =np.zeros((i.shape[0] + padding_h, i.shape[1] + padding_w))
        #     temp[:i.shape[0],:i.shape[1]] = i[:,:]
        #     padded_images.append(temp)
        #print("Train_index: ",trainIndex)


        blocks = sized//wiS
        chan = frameT.shape[1]
        

        for index,i in enumerate(images):
            # returns [ch,blocks,blocks,wiS,wiS]
            temp = as_strided(i,shape=(chan,blocks,blocks,wiS,wiS),strides=(8*prod(i.shape[1:]),blocks*wiS*8*wiS,wiS*8,blocks*wiS*8,8))
            temp = temp.reshape(chan,blocks*blocks,wiS,wiS)
            # Make DCT
            temp = scF.dctn(temp,axes=(2,3))
            temlen = temp.shape[1] * temp.shape[0]
            temp = temp.reshape(temlen,wiS*wiS)
            if(se_index + temlen >= stored_eighter.shape[0]-1):
                print("In here")
                temlen = stored_eighter.shape[0] - se_index
                stored_eighter[se_index:se_index+temlen,:] = temp[:temlen,:]
                se_index += temlen
            else:
                stored_eighter[se_index:se_index+temlen,:] = temp
                se_index += temlen

        if(se_index >= stored_eighter.shape[0]):
            break

    print("Now in the PCA step")
    pca = PCA()
    pca_transformed = pca.fit_transform(stored_eighter)
    pk.dump(pca,open("pca.pkl","wb"))
    #print(pca.components_[:5])
    total_var = sum(pca.explained_variance_)
    print(pca.explained_variance_ratio_[:20])



def main_trainset():
    params = Parameters(wiS=20,nSamples=2_000_000,on_gpu=False)
    print(params)
    data_train = X_Train(params)

    dataloader = torch.utils.data.DataLoader(data_train, batch_size=1, drop_last=True, shuffle=True,
                                             num_workers=4, pin_memory=False)

    create_pca(dataloader,params)


class X_Test(data.Dataset):
    def __init__(self, args, multiple, validation=False):
        self.args = args
        self.multiple = multiple
        self.validation = validation

        # Return """ make [I0,I1,It,t,scene_folder] """
        if validation:
            self.testPath = make_2D_dataset_X_Test(self.args.val_data_path, multiple, t_step_size=32)
        else:  ## test
            self.testPath = make_2D_dataset_X_Test(self.args.test_data_path, multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            if validation:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.val_data_path + "\n"))
            else:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.test_data_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]

        # Returns all frames (65)
        frames = frames_loader_test(self.args, I0I1It_Path, self.validation)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations

# Untouched
def make_2D_dataset_X_Test(dir, multiple, t_step_size):
    """ make [I0,I1,It,t,scene_folder] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for type_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [type1,type2,type3,...]
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
            frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # 32 multiple, ['00000.png',...,'00032.png']
            for idx in range(0, len(frame_folder), t_step_size):  # 0,32,64,...
                if idx == len(frame_folder) - 1:
                    break
                for mul in range(multiple - 1):
                    I0I1It_paths = []
                    I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                    I0I1It_paths.append(frame_folder[idx + t_step_size])  # I1 (fix)
                    I0I1It_paths.append(frame_folder[idx + int((t_step_size // multiple) * (mul + 1))])  # It
                    I0I1It_paths.append(t[mul])
                    I0I1It_paths.append(scene_folder.split(os.path.join(dir, ''))[-1])  # type1/scene1
                    testPath.append(I0I1It_paths)
    return testPath

def frames_loader_test(args, I0I1It_Path, validation=False):
    frames = []
    for path in I0I1It_Path:
        frame = cv2.imread(path)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)

    #print("Frame size test: ", frames.shape) [3,2160,4095,3]
    if args.dataset == 'X4K1000FPS':
        if validation:
            ps = args.validation_patch_size#512
        else:
            ps = args.test_patch_size
        if(ps != -1):
            ix = (iw - ps) // 2
            iy = (ih - ps) // 2
            frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    """ np2Tensor [-1,1] normalized """
    frames = my_RGBframes_np2Tensor(frames, args.img_ch,args.on_gpu)

    return frames

@dataclass
class DCTParamsAdap:
    weightMat = torch.zeros((2,2))
    wiS: int 
    gpu: int 
    components_fraction: float #= 1/64
    data_used: float #= 0.01
    h: int
    w: int


@dataclass
class DCTParams:
    weightMat = torch.zeros((2,2))
    wiS: int 
#    gpu: int 
    components_fraction: float #= 1/64
    data_used: float #= 0.01
#    h: int
#    w: int


    # def getH(self):
    #     return self.h//self.wiS + (0 if(self.h%self.wiS == 0) else 1)
    # def getW(self):
    #     return self.w//self.wiS + (0 if(self.w%self.wiS == 0) else 1)    

    # def getUBH(self):
    #     return (self.h//self.wiS + (0 if(self.h%self.wiS == 0) else 1))*self.wiS
    # def getUBW(self):
    #     return (self.w//self.wiS + (0 if(self.w%self.wiS == 0) else 1))*self.wiS

#to_dctpca(im,params,components_fraction=0)
#from_dctpca_to_dct_diff(res,eigenvectors,mean,params,pca)
#def to_dct(im,wiS):
#dct_inverse(im,wiS_param=0,params=0):

###############################         NEEDS TO BE FIXED, blocks_x and blocks_y           ###############################################
# input i.e. [8,16,20,20]
def pca_inverse(res,params,pcas,comps_used,cut_back=True,wanted_dim=0,args=0):
    #print(eigenvectors.shape)
    B,comps,BY,BX = res.shape

    eigenvectors = torch.cat([torch.as_tensor(i.eigenvectors[:comps_used,:],device=params.gpu).unsqueeze(0) for i in pcas])#torch.as_tensor(pca.eigenvectors[:comps_used,:],device=params.gpu)
    mean = torch.cat([torch.as_tensor(i.mean,device=params.gpu).unsqueeze(0) for i in pcas],dim=0).unsqueeze(1)
    wiS = params.wiS
    sized = wiS**2
    
    temp = ((res+1)/2)
    if(not args.maxmin_vec):
        for i in range(len(pcas)):
            mi,ma = pcas[i].store["mima"]
            if(args.weightMat):
                res = res.permute((0,2,3,1)).reshape(-1,comps)
                res = torch.matmul()
                # BITTE HIER WEITER MACHEN! batched matmul und backtransform to original view
            temp[i,:] = temp[i,:]*(ma - mi) + mi


 
    #print("mean: ",mean.shape)
    temp = temp.permute((0,2,3,1))
    temp = temp.reshape(B,-1,comps)
    temp = temp.reshape(B,-1,3,comps//3)
    temp = temp.reshape(B,-1,comps//3).double()

    # Two normalization Concepts!!!!!!!!
    if(args.mean_vector_norm):
        #print("temp shape: ",temp.shape)
        for i in range(len(pcas)):
            comps_used = pcas[i].store["comps_used"]
            mean_vec = pcas[i].store["mean_vec"]
            temp_shap = temp[i,:].shape
            temp[i,:] = (temp[i,:]* mean_vec[:temp_shap[-1]])#.reshape(temp_shap)
    if(args.maxmin_vec):
        for i in range(len(pcas)):
            mi,ma = pcas[i].store["mima"]
            ma = ma[:temp[i,:].shape[-1]]
            mi = mi[:temp[i,:].shape[-1]]
            #print("mi ma: ",mi,ma)
            temp[i,:] = (temp[i,:] *(ma-mi)) + mi

    # back unnormalization          res = (((res - mi)/(ma-mi))*2)-1
    
    cupy_pca = False
    # Components to actual DCT
    if(cupy_pca):
        #print("shape before pca: ",temp.shape)
        temp = cp.asarray(temp)#temp.cpu().numpy()
        back = pca.inverse_transform(temp)
        back = torch.tensor(back,device=params.gpu)
    else:
        #print(temp.shape,eigenvectors.shape)
        back = torch.matmul(temp,eigenvectors)
        #print(back.shape,mean.shape)
        back += mean

    # Back transformation to Right format
    back = back.reshape(B,-1,3,sized).reshape(B,BY,BX,3,sized).permute((0,3,1,2,4)).reshape(B,3,BY,BX,wiS,wiS).permute((0,1,3,2,4,5)) # now: [C,blockx,blocky,wiS,wiS]
    back = back.reshape(B,3,BX,wiS*BY,wiS)
    back = torch.cat([ back[:,:,i,:,:]for i in range(back.shape[2])],axis=3)
    if(cut_back):
        back = back[:,:,:wanted_dim[0],:wanted_dim[1]]

    return back

# IM: [C,H,W]
def to_pca(im,params,components_fraction=0,args=0,pca=0):
    #print(params)
    mempool = cp.get_default_memory_pool()
    wiS = params.wiS
    weightMat = params.weightMat
    if(components_fraction == 0):
        components_used = int((wiS*wiS)*params.components_fraction)
    else:
        components_used = int((wiS*wiS)*components_fraction)

    data_used = params.data_used
    chan = im.shape[0]
    height = im.shape[1]
    width = im.shape[2]
    pad_y = wiS - height%wiS #height - (height//wiS)*wiS
    pad_x = wiS - width%wiS #width - (width//wiS)*wiS
    if(pad_y == wiS):
        pad_y = 0
    if(pad_x == wiS):
        pad_x = 0
    pad_x =0
    pad_y = 0
    blocks_y = height//wiS +(1 if(pad_y >0)else 0)
    blocks_x = width//wiS +(1 if(pad_x >0)else 0)

    image = np.zeros((chan,height + pad_y, width + pad_x))
    #image[:,:height,:width] = im[:,:,:]
    
    image[:,:,:] = np.pad(im,((0,0),(0,pad_y),(0,pad_x)) ,mode=args.padding)
    im = 0
    #print("My big blocked shape BEFORE!!!!!!!!: ",image.shape)
    # Into blocks
    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    #print("My big blocked shape!!!!!!!!: ",blocked.shape)


    #blocked = scF.dctn(blocked,axes=(3,4))    #blocked_gpu = cufft.fftn(blocked_gpu,axes=(3,4)).real 
    #CUPY FFT
    pca_ready = blocked.reshape(-1,wiS*wiS)


    # GPU########################################
    #pca_ready_gpu = cp.asarray(pca_ready)
    pca_ready_gpu = torch.as_tensor(pca_ready,dtype=torch.float64,device=args.gpu)
    #print("PCA READY GPU: ",pca_ready_gpu.shape)

    # choice = False
    # if(not choice):
    in1 = pca_ready_gpu # cp.asarray(pca_ready_gpu.clone())
    if(args.weightMat):
        learnedPCA = MYPCA(n_components=wiS**2)#(n_components=wiS**2)
    else:
        learnedPCA = MYPCA(n_components=components_used)

    #print("in1: ",in1.shape)
    if(pca==0):
        res = learnedPCA.fit_transform(in1,args.gpu)
    else:
        #res = learnedPCA.fit_transform(in1)#[:,:components_used]
        res = pca.transform(in1,args.gpu)
    #print("res: ",res.shape)
    #res = torch.as_tensor(res,device=args.gpu)
        

    
    if(args.mean_vector_norm):
        if(pca == 0):
            mean_vec = torch.mean(torch.abs(res),dim=0)
            learnedPCA.store_sth(mean_vec,"mean_vec")
            learnedPCA.store_sth(components_used,"comps_used")
        else:
            mean_vec = pca.store["mean_vec"].to(args.gpu)
        res = res/mean_vec
        
    if(args.maxmin_vec):
        if(pca==0):
            ma = torch.max(res,axis=0)[0]
            mi = torch.min(res,axis=0)[0]
            learnedPCA.store_sth((mi,ma),"mima")
        else:
            mi,ma = pca.store["mima"]
        #print("mi ma: ",mi,ma)
        res = (res - mi)/(ma - mi)
        

    # res: (n,comps) comps=wiS**2
    # W: (comps,wiS**2)
    # ret: (n,comps)
    if(args.weightMat):
        res = torch.matmul(weightMat,res.T).T
        #print(params)
        #print(params.weightMat.shape)
        #print("Result shape of weightmat: ",res.shape)
        
        

    res = res.reshape(chan,blocks_x,blocks_y,components_used)    
    res = res.permute((0,3,2,1))#cp.transpose(res,(0,3,1,2))
    res = res.reshape(-1,blocks_y,blocks_x) # chan * components,blocks_y,blocks_x
    
    # Normalize it!
    if(not args.maxmin_vec):
        if(pca==0):
            mi = torch.min(res)
            ma = torch.max(res)
            learnedPCA.store_sth((mi,ma),"mima")
        else:
            mi,ma = pca.store["mima"]
        res = (((res - mi)/(ma-mi)))

    # to [-1,1]
    res = (res*2)-1
    #mi = torch.as_tensor(mi,device=params.gpu)
    #ma = torch.as_tensor(ma,device=params.gpu)
    
    mempool.free_all_blocks()
    
    return res,learnedPCA

# im: [6,H,W]
def to_pca_diff(im,params,args,mean,EV,mean_vec):
    #print(params)
    start_time = time.time()
    wiS = params.wiS
    weightMat = params.weightMat
    components_used = int((wiS*wiS)*params.components_fraction)

    chan = im.shape[0]
    height = im.shape[1]
    width = im.shape[2]
    
    #print(pad_x,pad_y)
    blocks_y = height//wiS 
    blocks_x = width//wiS 
    if(height % wiS != 0 or width % wiS != 0):
        raise Exception("in to_pca_diff the image is not padded right." + str(height)+" "+str(width))

    if(False): # efficient way
        image = np.zeros(im.shape)
        image[:] = im.to("cpu").detach().numpy()
        # Into blocks
        blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
        #56.623.104
    else:
        #print("blocked shape: ",blocked.shape)
        #start_time = time.time()
        imtem = torch.as_tensor(im).unsqueeze(0)
        toblockfun = nn.Unfold(kernel_size=wiS,stride=wiS)
        #print("Imtem: ",imtem.shape)
        imtem = toblockfun(imtem).squeeze(0)
        blocked = imtem.reshape(-1,blocks_y,blocks_x).permute(0,2,1).reshape(chan,wiS**2,blocks_x,blocks_y).permute(0,2,3,1).reshape(chan,blocks_x,blocks_y,wiS,wiS)   #.reshape(6,wiS,wiS,blocks_y,blocks_x).permute(0,4,3,2,1)
        # [C*wiS**2,L]
        #print("after transf: ",imtem.shape)
        #print("time for pca blockwise: ",time.time()-start_time)
        #blocked_test = torch.as_tensor(blocked)
        #print("part error: ",(processedimtem[:,:,:,0]-blocked_test[:,:,:,0,0]).abs().sum())
        #print("error: ",(processedimtem.cpu()-blocked_test).abs().sum())




    pca_ready_gpu = torch.as_tensor(blocked.reshape(-1,wiS*wiS),device=args.gpu)


    #res = pca.transform(in1)
    loc_data = pca_ready_gpu - mean
    assert not pca_ready_gpu.isnan().any(), "pca ready gpu"
    assert not mean.isnan().any(), "mean"
    assert not EV.isnan().any(), "EV"

    if(args.timetest):
        print("PCA preparation: ",time.time()-start_time)
    start_time = time.time()
    transformed = torch.matmul(loc_data,EV.permute(1,0))
    assert not transformed.isnan().any(), "transformed"
    
    if(args.timetest):
        print("Matmul: ",time.time()-start_time)
    start_time = time.time()
    if(args.mean_vector_norm):
        transformed = transformed/mean_vec
    
    assert not transformed.isnan().any(), "after meanvec transformed"


    transformed = transformed.reshape(chan,blocks_x,blocks_y,components_used)    
    transformed = transformed.permute((0,3,2,1))#cp.transpose(res,(0,3,1,2))
    transformed = transformed.reshape(-1,blocks_y,blocks_x) # chan * components,blocks_y,blocks_x
    
    # Normalize it!
    if(not args.maxmin_vec):
        mi = torch.min(transformed)
        ma = torch.max(transformed)
        transformed = (((transformed - mi)/(ma-mi)))

    # to [-1,1]
    transformed = (transformed*2)-1
    
    #print("end of diff: ",transformed.shape)
    if(args.timetest):
        print("Rest of PCA: ",time.time()-start_time)
    return transformed

def pca_inverse_testing(res,params,pcas,comps_used,cut_back=True,wanted_dim=0):
    #print(eigenvectors.shape)
    B,comps,BY,BX = res.shape
    eigenvectors = torch.cat([torch.as_tensor(i.eigenvectors[:comps_used,:],device="cpu").unsqueeze(0) for i in pcas])#torch.as_tensor(pca.eigenvectors[:comps_used,:],device=params.gpu)
    mean = torch.cat([torch.as_tensor(i.mean,device="cpu").unsqueeze(0) for i in pcas],dim=0).unsqueeze(1)
    wiS = params.wiS
    sized = wiS**2
    
    

 
    #print("mean: ",mean.shape)
    temp = res.permute((0,2,3,1))
    temp = temp.reshape(B,-1,comps)
    temp = temp.reshape(B,-1,3,comps//3)
    temp = temp.reshape(B,-1,comps//3).double()

    
    back = torch.matmul(temp,eigenvectors)
    #print(back.shape,mean.shape)
    back += mean

    # Back transformation to Right format
    back = back.reshape(B,-1,3,sized).reshape(B,BY,BX,3,sized).permute((0,3,1,2,4)).reshape(B,3,BY,BX,wiS,wiS).permute((0,1,3,2,4,5)) # now: [C,blockx,blocky,wiS,wiS]
    back = back.reshape(B,3,BX,wiS*BY,wiS)
    back = torch.cat([ back[:,:,i,:,:]for i in range(back.shape[2])],axis=3)
    if(cut_back):
        back = back[:,:,:wanted_dim[0],:wanted_dim[1]]

    return back

def to_pca_testing(im,params,components_fraction=0,pca=0,compsused=0):
    #print(params)

    wiS = params.wiS
    weightMat = params.weightMat
    if(components_fraction == 0):
        components_used = int((wiS*wiS)*params.components_fraction)
    else:
        components_used = int((wiS*wiS)*components_fraction)
    data_used = params.data_used

    chan = im.shape[0]
    height = im.shape[1]
    width = im.shape[2]
    pad_y = wiS - height%wiS #height - (height//wiS)*wiS
    pad_x = wiS - width%wiS #width - (width//wiS)*wiS
    if(pad_y == wiS):
        pad_y = 0
    if(pad_x == wiS):
        pad_x = 0
    
    #print(pad_x,pad_y)
    blocks_y = height//wiS +(1 if(pad_y >0)else 0)
    blocks_x = width//wiS +(1 if(pad_x >0)else 0)

    image = np.zeros((chan,height + pad_y, width + pad_x))
    #image[:,:height,:width] = im[:,:,:]
    
    image[:,:,:] = np.pad(im,((0,0),(0,pad_y),(0,pad_x)) ,mode="reflect")
    im = 0
    #print("My big blocked shape BEFORE!!!!!!!!: ",image.shape)
    # Into blocks
    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    #print("My big blocked shape!!!!!!!!: ",blocked.shape)


    #blocked = scF.dctn(blocked,axes=(3,4))    #blocked_gpu = cufft.fftn(blocked_gpu,axes=(3,4)).real 
    #CUPY FFT
    pca_ready = blocked.reshape(-1,wiS*wiS)


    # GPU########################################
    #pca_ready_gpu = cp.asarray(pca_ready)
    pca_ready_gpu = torch.as_tensor(pca_ready,dtype=torch.float64,device="cpu")
    #print("PCA READY GPU: ",pca_ready_gpu.shape)

    # choice = False
    # if(not choice):
    in1 = pca_ready_gpu#cp.asarray(pca_ready_gpu.clone())
    #pca = MYPCA(n_components=components_used)
    #print("in1: ",in1.shape)
    if(pca==0):
        learnedPCA = MYPCA(n_components=components_used)
        learnedPCA.fit(in1)
        res = in1
    else:
        learnedPCA = pca
        if(compsused==0):
            res = pca.transform(in1,device="cpu")#[:,:components_used]
        else:
            res = pca.transform(in1,device="cpu",compsused=compsused)

    #print("res: ",res.shape)
        #res = torch.as_tensor(res,device=params.gpu)
            

        res = res.reshape(chan,blocks_x,blocks_y,components_used)    
        res = res.permute((0,3,2,1))#cp.transpose(res,(0,3,1,2))
        res = res.reshape(-1,blocks_y,blocks_x) # chan * components,blocks_y,blocks_x
    
    return res,learnedPCA

import datetime
def test_on_dataset(dataloader,params):
    wiS = params.wiS
    log_file = open( 'ownTextFiles/PCA-Generalization.txt', 'w')
    log_file.write('----- PCA Generalization -----\n')
    log_file.write(str(datetime.datetime.now())[:-7] + '\n')
    
    upanddown = False

    for i in range(10):
        count = 0
        psnrlist = []
        pca = 0
        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(dataloader):
            #frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            input_frames = (frames[:, :, :-1, :, :]+1)/2    #([1, 3, 2, 2160, 4096]) 
            input_frames = input_frames[0,:,0,:1080,:2048].cpu() # to [C,H,W]
            
            #print(input_frames.shape)
            temp_pic = input_frames.clone().numpy()
            if(upanddown):
                input_frames = nn.functional.interpolate(input_frames.unsqueeze(0),scale_factor=2,mode="nearest").squeeze(0)
            #print(input_frames.shape)
            
            #input_frames = input_frames[:,:1024,:1024]
            #print("max, min: ",torch.max(input_frames).item(),torch.min(input_frames).item())

            params = DCTParamsAdap(wiS=8,gpu=1,components_fraction=1/64,data_used=0.01,h=input_frames.shape[1],w=input_frames.shape[2])
            if(count == 0):
                print(params)
            



            if(count == 0):
                _,pca = to_pca_testing(input_frames,params,components_fraction=0)
                #print("This frame it is learned from: ",scene_name,frameRange)
                log_file.write("This frame it is learned from: " + scene_name[0] +" \nPic name: "+frameRange[0][0] + "\n")
                #thepca = pca
            #print("before to pca_tesing: ",input_frames.shape)
            res,_ = to_pca_testing(input_frames,params,components_fraction=0,pca=pca)
            #res = res.unsqueeze(0)
            #print("after ",res.shape)
            res = torch.as_tensor(res,device="cpu").unsqueeze(0)
            #print(res.shape)
            #torch_prints(res,"PCA INFO")
            #print("intermediate shape: ",res.shape)
            #res,params,pcas,comps_used,cut_back=True,wanted_dim=0
            reconstructed = pca_inverse_testing(res,params,[pca],comps_used=int(wiS**2 *params.components_fraction),cut_back=True,wanted_dim=input_frames.shape[-2:])
            
            if(upanddown):
                reconstructed = nn.functional.interpolate(reconstructed,scale_factor=1/2,mode="bilinear").cpu().numpy().squeeze(0)
            else:
                reconstructed = reconstructed.squeeze(0).cpu().numpy()

            #reconstructed = dct_inverse(reconstructed,params=params)
            # DCT and back
            # res = to_dct(input_frames,wiS=params.wiS)
            # input_frames = dct_inverse(res,params=params)
            # # DCTPCA and Back
            # dctpcaT,pca = to_dctpca(input_frames,params)
            # dctpcaT = torch.as_tensor(dctpcaT,device=params.gpu).unsqueeze(0) # to tensor and batchsize added (1)
            # reconstructed = from_dctpca_to_dct_diff(dctpcaT,params,pca).cpu().numpy().squeeze(0)
            # reconstructed = dct_inverse(reconstructed,params=params)


            #print("Reconstructed shape: ",reconstructed.shape)
            #print("absolute differences: ",np.sum(np.abs(temp_pic-reconstructed)))

            #print(temp_pic.shape,reconstructed.shape)
            #numpy_prints(reconstructed,"reconstructed")
            #numpy_prints(temp_pic,"temp")

            psnrResult = peak_signal_noise_ratio(temp_pic,reconstructed,data_range=1)
            if(count == 0):
                log_file.write("PSNR of original frame: " + str(psnrResult) + "\n")
                print("PSNR of original frame: ",psnrResult)
            psnrlist.append(psnrResult)
             #ret is numpy
            
            #print("Reconstructed max, min: ",np.max(reconstructed),np.min(reconstructed))
            #cv2.imwrite("tempTest/temppic.png",np.transpose(temp_pic,[1, 2, 0])*255)
            #cv2.imwrite("tempTest/retpic.png",np.transpose(reconstructed,[1, 2, 0])*255)
            count += 1
            if(count == 50):
                break
        
        #print("PSNR LIST: ",psnrlist)
        psnrlist = np.array(psnrlist[1:])
        print("PSNR mean: ",np.mean(psnrlist))
        print("PSNR std: ",np.std(psnrlist))
        log_file.write("PSNR mean: "+ str(np.mean(psnrlist)) + "\n")
        log_file.write("PSNR std: "+ str(np.std(psnrlist))+ "\n")

    
    log_file.close()


import torch.nn.functional as F
def reconstruction_test_scales(dataloader,params):

    once_computed = True
    use_learned = True
    if use_learned:
        checkpoint = torch.load(os.path.join("ownTextFiles" ,"XVFInet_X4K1000FPS_exp2059_best_PSNR.pt"))
        # [16,64]
        ev8 = checkpoint["state_dict_Model"]["EV8"] 
        # [64]
        mean8 = checkpoint["state_dict_Model"]["Mean8"]
        learned_pca = MYPCA()
        learned_pca.mean = mean8
        learned_pca.eigenvectors = ev8
        print(ev8)
        print(mean8)

    wiS = params.wiS
    log_file = open( 'ownTextFiles/PCA-Reconstruction.txt', 'w')
    log_file.write('----- PCA Reconstruction -----\n')
    log_file.write(str(datetime.datetime.now())[:-7] + '\n')


    scales =[1,1/4,1/16,1/64] #[1,1/4,1/16,1/64]
    possib = []
    for i in scales:
        for k in [1/4,1/8,1/16,1/32,1/64]:
            possib.append([i,k])
    # possib = []
    # for i in b:
    #     for k in range(int(math.log2(i**2))):
    #         possib.append([i,1/(2**(k+1))])

    counting = 0
    once = True
    for i in possib:
        psnrlist = []
        pca = 0
        log_file.write('----- Scale: {}  Fractions: {} -----\n'.format(i[0],i[1]))
        print('----- Scale: {}  Fractions: {} -----\n'.format(i[0],i[1]))
        
        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(dataloader):

            if(not use_learned ):
                input_frames = (frames[:, :, :-1, :, :]+1)/2    #([1, 3, 2, 2160, 4096])
            else:
                input_frames = frames[:, :, :-1, :, :]

            input_frames = input_frames[0,:,0,:,:].cpu() # to [C,H,W]  #824:1336,1792:2304
            if(i[0]<1):
                input_frames = F.interpolate(input_frames.unsqueeze(0) ,scale_factor=i[0],mode="bilinear").squeeze(0)
            #print(input_frames.shape)
           


            #print(input_frames.shape)
            temp_pic = input_frames.clone().numpy()
            #print(input_frames.shape)

            #input_frames = input_frames[:,:1024,:1024]
            #print("max, min: ",torch.max(input_frames).item(),torch.min(input_frames).item())

            params = DCTParamsAdap(wiS=8,gpu=1,components_fraction=1/4,data_used=0.01,h=input_frames.shape[1],w=input_frames.shape[2])

            if(use_learned):
                res,_ = to_pca_testing(input_frames,params,components_fraction=0,pca=learned_pca,compsused=int(i[1]*(8**2)))
            elif(once_computed):
                if(once):
                    _,pca = to_pca_testing(input_frames,params,components_fraction=0)
                    once = False
                    continue
                else:
                    res,_ = to_pca_testing(input_frames,params,components_fraction=0,pca=pca,compsused=int(i[1]*(8**2)))
            else:
                _,pca = to_pca_testing(input_frames,params,components_fraction=0)
                res,_ = to_pca_testing(input_frames,params,components_fraction=0,pca=pca)
            #res = res.unsqueeze(0)
            #print("after ",res.shape)
            res = torch.as_tensor(res,device="cpu").unsqueeze(0)
            #print(res.shape)
            #torch_prints(res,"PCA INFO")
            #print("intermediate shape: ",res.shape)
            #res,params,pcas,comps_used,cut_back=True,wanted_dim=0
            if(use_learned):
                reconstructed = pca_inverse_testing(res,params,[learned_pca],comps_used=int(i[1]*(8**2)),cut_back=True,wanted_dim=input_frames.shape[-2:])
                reconstructed = (reconstructed +1)/2
                temp_pic = (temp_pic +1)/2
            else:
                reconstructed = pca_inverse_testing(res,params,[pca],comps_used=int(i[1]*(8**2)),cut_back=True,wanted_dim=input_frames.shape[-2:])
            reconstructed = reconstructed.squeeze(0).cpu().numpy()

            cv2.imwrite("ownTextFiles/"+str(testIndex)+".png",np.transpose(reconstructed,(1,2,0))*255)
            #cv2.imwrite("ownTextFiles/"+str(testIndex)+".png",np.transpose(temp_pic,(1,2,0))*255)
            
            psnrResult = peak_signal_noise_ratio(temp_pic,reconstructed,data_range=1)
            print("PSNR Result: ",psnrResult)
            psnrlist.append(psnrResult)
        break
        #print("Counting: ",counting)
        #print("PSNR LIST: ",psnrlist)
        psnrlist = np.array(psnrlist)
        print("PSNR mean: ",np.mean(psnrlist))
        print("PSNR std: ",np.std(psnrlist))
        log_file.write("PSNR mean: "+ str(np.mean(psnrlist)) + "\n")
        log_file.write("PSNR std: "+ str(np.std(psnrlist))+ "\n" + "\n")


    log_file.close()


def reconstruction_test(dataloader,params):

    once_computed = True
    use_learned = False
    # if use_learned:
    #     checkpoint = torch.load(os.path.join("ownTextFiles" ,"XVFInet_X4K1000FPS_exp2059_best_PSNR.pt"))
    #     # [16,64]
    #     ev8 = checkpoint["state_dict_Model"]["EV8"] 
    #     # [64]
    #     mean8 = checkpoint["state_dict_Model"]["Mean8"]
    #     learned_pca = MYPCA()
    #     learned_pca.mean = mean8
    #     learned_pca.eigenvectors = ev8

    wiS = params.wiS
    log_file = open( 'ownTextFiles/PCA-Reconstruction.txt', 'w')
    log_file.write('----- PCA Reconstruction -----\n')
    log_file.write(str(datetime.datetime.now())[:-7] + '\n')

    b = [4,8,16,32,64]
    #scales = [1,1/2,1/4,1/8,1/16]
    possib = []
    for i in b:
        for k in range(int(math.log2(i**2))):
            possib.append([i,1/(2**(k+1))])

    counting = 0
    for i in possib:
        psnrlist = []
        pca = 0
        once = True
        log_file.write('----- Blocksize: {}  Fractions: {} -----\n'.format(i[0],i[1]))
        print('----- Blocksize: {}  Fractions: {} -----\n'.format(i[0],i[1]))

        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(dataloader):

            input_frames = (frames[:, :, :-1, :, :]+1)/2    #([1, 3, 2, 2160, 4096])
            input_frames = input_frames[0,:,0,824:1336,1792:2304].cpu() # to [C,H,W]  #824:1336,1792:2304
            #print(input_frames.shape)
            assert input_frames.shape == (3,512,512)


            #print(input_frames.shape)
            temp_pic = input_frames.clone().numpy()
            #print(input_frames.shape)

            #input_frames = input_frames[:,:1024,:1024]
            #print("max, min: ",torch.max(input_frames).item(),torch.min(input_frames).item())

            params = DCTParamsAdap(wiS=i[0],gpu=1,components_fraction=i[1],data_used=0.01,h=input_frames.shape[1],w=input_frames.shape[2])

            
            if(once_computed):
                if(once):
                    _,pca = to_pca_testing(input_frames,params,components_fraction=0)
                    once = False
                    continue
                else:
                    res,_ = to_pca_testing(input_frames,params,components_fraction=0,pca=pca)
            else:
                _,pca = to_pca_testing(input_frames,params,components_fraction=0)
                res,_ = to_pca_testing(input_frames,params,components_fraction=0,pca=pca)
            #res = res.unsqueeze(0)
            #print("after ",res.shape)
            res = torch.as_tensor(res,device="cpu").unsqueeze(0)
            #print(res.shape)
            #torch_prints(res,"PCA INFO")
            #print("intermediate shape: ",res.shape)
            #res,params,pcas,comps_used,cut_back=True,wanted_dim=0
            reconstructed = pca_inverse_testing(res,params,[pca],comps_used=int((i[0]**2)*i[1]),cut_back=True,wanted_dim=input_frames.shape[-2:])
            reconstructed = reconstructed.squeeze(0).cpu().numpy()


            psnrResult = peak_signal_noise_ratio(temp_pic,reconstructed,data_range=1)

            psnrlist.append(psnrResult)

        #print("Counting: ",counting)
        #print("PSNR LIST: ",psnrlist)
        psnrlist = np.array(psnrlist)
        print("PSNR mean: ",np.mean(psnrlist))
        print("PSNR std: ",np.std(psnrlist))
        log_file.write("PSNR mean: "+ str(np.mean(psnrlist)) + "\n")
        log_file.write("PSNR std: "+ str(np.std(psnrlist))+ "\n" + "\n")


    log_file.close()

@dataclass
class Parameters:
    nSamples: int
    wiS: int 
    on_gpu: bool = True
    multiple: int = 8
    train_path: str = './../../../' + 'X-Train/' + 'train'
    val_data_path: str = './../../../' + 'X-Train/' + 'val'
    test_data_path: str = './../../../' + 'X-Train/' + 'test' 
    dataset: str =  'X4K1000FPS'
    test_patch_size: int = -1
    need_patch: bool = False
    patch_size: int = 384
    img_ch: int = 3
    max_t_step_size: int = 32
    components_fraction: float = 1/64
    data_used = 0.01

def main_testset():
    params = Parameters(wiS=20,nSamples=2_000_000,on_gpu=False)
    #print(params)
    data_test = X_Test(params,params.multiple)

    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, drop_last=True, shuffle=False,
                                             num_workers=2, pin_memory=False)

    reconstruction_test(dataloader,params)

if __name__ == "__main__":
    main_testset()
