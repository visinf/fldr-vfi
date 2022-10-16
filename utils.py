from __future__ import division
from asyncio import FastChildWatcher
import os, glob, sys, torch, shutil, random, math, time, cv2
from tracemalloc import start
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from torch.nn import init
#from skimage.measure import compare_ssim
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from torch.autograd import Variable
from torchvision import models

from myAdditions import torch_prints
#from pca_comp import pca_trans_and_back

from skimage.transform import rescale

from yuv_frame_io import YUV_Read,YUV_Write
from softSplat import Softsplat
from PIL import Image
class save_manager():
    def __init__(self, args):
        self.args = args

        # reused directory name for this model
        self.model_dir = self.args.net_type + '_' + self.args.dataset + '_exp' + str(self.args.exp_num)
        print("model_dir:", self.model_dir)
        # ex) model_dir = "XVFInet_exp1"

        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)
        # './checkpoint_dir/XVFInet_exp1"
        check_folder(self.checkpoint_dir)

        print("checkpoint_dir:", self.checkpoint_dir)

        self.text_dir = os.path.join(self.args.text_dir, self.model_dir)
        print("text_dir:", self.text_dir)

        """ Save a text file """
        if (not os.path.exists(self.text_dir + '.txt') or not args.continue_training) and not args.phase=='test':
            self.log_file = open(self.text_dir+ '.txt', 'w')
            # "w" - Write - Opens a file for writing, creates the file if it does not exist
            self.log_file.write('----- Model parameters -----\n')
            self.log_file.write(str(datetime.now())[:-7] + '\n')
            for arg in vars(self.args):
                self.log_file.write('{} : {}\n'.format(arg, getattr(self.args, arg)))
            # ex) ./text_dir/XVFInet_exp1.txt
            self.log_file.close()

    # "a" - Append - Opens a file for appending, creates the file if it does not exist

    def write_info(self, strings):
        self.log_file = open(self.text_dir+ '.txt', 'a')
        self.log_file.write(strings)
        self.log_file.close()

    def save_best_model(self, combined_state_dict, best_PSNR_flag):
        file_name = os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt')
        # file_name = "./checkpoint_dir/XVFInet_exp1/XVFInet_exp1_latest.ckpt
        torch.save(combined_state_dict, file_name)
        if best_PSNR_flag:
            shutil.copyfile(file_name, os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'))

    # file_path = "./checkpoint_dir/XVFInet_exp1/XVFInet_exp1_best_PSNR.ckpt

    def save_epc_model(self, combined_state_dict, epoch):
        file_name = os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch) + '.pt')
        # file_name = "./checkpoint_dir/XVFInet_exp1/XVFInet_exp1_epc10.ckpt
        torch.save(combined_state_dict, file_name)

    def load_epc_model(self, epoch):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch - 1) + '.pt'))
        print("load model '{}', epoch: {}, best_PSNR: {:3f}".format(
            os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch - 1) + '.pt'), checkpoint['last_epoch'] + 1,
            checkpoint['best_PSNR']))
        return checkpoint

    def load_model(self,specific=-1 ,takeBestModel=False):
        # checkpoint = torch.load(self.checkpoint_dir + '/' + self.model_dir + '_latest.pt')
        suffix =  '_latest.pt'
        if(specific != -1):
            suffix = '_epc' + str(specific) + '.pt'
        elif(takeBestModel):
            suffix = '_best_PSNR.pt'
        print(os.path.join(self.checkpoint_dir, self.model_dir +suffix))
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + suffix), map_location="cuda:"+str(self.args.gpu))
        #print("I am here")
        print("load model '{}', epoch: {},".format(
            os.path.join(self.checkpoint_dir, self.model_dir + suffix), checkpoint['last_epoch'] + 1))
        return checkpoint

    def load_best_PSNR_model(self, ):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'))
        print("load _best_PSNR model '{}', epoch: {}, best_PSNR: {:3f}, best_SSIM: {:3f}".format(
            os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'), checkpoint['last_epoch'] + 1,
            checkpoint['best_PSNR'], checkpoint['best_SSIM']))
        return checkpoint


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir


def weights_init(m):
    classname = m.__class__.__name__
    if (classname.find('Conv2d') != -1) or (classname.find('Conv3d') != -1):
        init.xavier_normal_(m.weight)
        # init.kaiming_normal_(m.weight, nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def get_train_data(args, max_t_step_size,device,genseed=None,shuffle=True):
    if args.dataset == 'X4K1000FPS':
        data_train = X_Train(args, max_t_step_size,device,deter=not shuffle)
    elif args.dataset == 'Vimeo':
        data_train = Vimeo_Train(args)
    elif args.dataset == "Inter4K":
    	data_train = Inter4K_Train(args,max_t_step_size=args.inter4k_stepsize)
    if(genseed == None):
        dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, drop_last=True, shuffle=shuffle,
                                             num_workers=int(args.num_thrds), pin_memory=args.pin_memory_train,prefetch_factor=3)
    else:
        dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, drop_last=True, shuffle=shuffle,
                                             num_workers=int(args.num_thrds), pin_memory=args.pin_memory_train,prefetch_factor=3,worker_init_fn=seed_worker,generator=g)
    
    return dataloader,(None,None)#(data_train.psnr_dct,data_train.psnr_bil)


def get_test_data(args,dataset, multiple, validation,specific=""):
    #dataset = args.dataset 
    if(len(specific)>0):
        dataset = specific
    if dataset == 'X4K1000FPS' and args.phase != 'test_custom':
        data_test = X_Test(args, multiple, validation)  # 'validation' for validation while training for simplicity
    elif dataset == 'XTest2KC' and args.phase != 'test_custom':
        data_test = X_Test(args, multiple, validation,twoKC=True) 
    elif dataset == 'Vimeo' and args.phase != 'test_custom':
        data_test = Vimeo_Test(args, validation)
    elif dataset == 'test_custom':
        data_test = Custom_Test(args, multiple)
    elif dataset == "Xiph" :
        data_test = Xiph_Test(args,validation)
    elif dataset == "Xiph2KC" :
        data_test = Xiph_Test(args,validation,twoKC=True)
    elif dataset == "Adobe240":
        data_test = Adobe_Test(args)
    elif dataset == "Inter4K88":
        data_test = Inter4K_Test(args,multiple,scenerange=8) #Inter4K_TestExtreme(args,multiple) 
    elif dataset == "Inter4K816":
        data_test = Inter4K_Test(args,multiple,scenerange=16) #Inter4K_TestExtreme(args,multiple) 
    elif dataset == "HD":
        data_test = HD_Test(args,multiple)
    
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, drop_last=False, shuffle=False, pin_memory=args.pin_memory_test)
    return dataloader

class Adobe_Test(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.framesPath = []
        folders = os.listdir(self.args.adobe_data_path)
        glob.glob(".png")
        # folder_name/GT/__.jpg
        #for i in folders:
        for fold in folders:
            file_names = sorted(glob.glob((os.path.join(self.args.adobe_data_path,fold, "GT", "*.jpg"))))
            if(len(file_names) <1):
                continue

            ran_f = min(len(file_names),24)
            for i in range(7):
                #print(int((i+1)/8 * len(file_names)))
                #print(file_names[0])
                self.framesPath.append([file_names[0],file_names[ran_f-1],file_names[int((i+1)/8 * ran_f)],(i+1)/8])

       

        #sys.exit()
        self.num_scene = len(self.framesPath)  # total test scenes
        if len(self.framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.xiph_data_path + "\n"))
        else:
            print("# of Adobe240 triplet testset : ", self.num_scene)

    def __getitem__(self, idx):
        scene_name = self.framesPath[idx][0].split(os.sep)
        scene_name = os.path.join(scene_name[-3], scene_name[-2],scene_name[-1].split(".png")[0])
        I0, I1, It, t_value = self.framesPath[idx]

        I0I1It_Path = [I0, I1, It]
        frames = frames_loader_test(self.args, I0I1It_Path, validation=False,Xiph=True)
        
        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        if(list(frames.shape) == [3,3,1280,720]):
            frames = frames.permute(0,1,3,2) 
        if(list(frames.shape) == [3,3,1920,1080]):
            frames = frames[:,:,:1920,:720].permute(0,1,3,2)
        if(list(frames.shape) == [3,3,1080,1920]):
            frames = frames[:,:,:720,:1280]


        for i in frames:
            assert list(i.shape) == [3,720,1280] , i.shape
          #print("frame: ",i.shape)

        print(frames.shape)
        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.num_scene


class HD_Test(data.Dataset):
    def __init__(self, args,multiple):
        self.args = args
        self.multiple = multiple
        self.framesPath = [
            (os.path.join(self.args.hd_data_path,'HD720p_GT/parkrun_1280x720_50.yuv'), 720, 1280),
            (os.path.join(self.args.hd_data_path,'HD720p_GT/shields_1280x720_60.yuv'), 720, 1280),
            (os.path.join(self.args.hd_data_path,'HD720p_GT/stockholm_1280x720_60.yuv'), 720, 1280),
            (os.path.join(self.args.hd_data_path,'HD1080p_GT/BlueSky.yuv'), 1080, 1920),
            (os.path.join(self.args.hd_data_path,'HD1080p_GT/Kimono1_1920x1080_24.yuv'), 1080, 1920),
            (os.path.join(self.args.hd_data_path,'HD1080p_GT/ParkScene_1920x1080_24.yuv'), 1080, 1920),
            (os.path.join(self.args.hd_data_path,'HD1080p_GT/sunflower_1080p25.yuv'), 1080, 1920),
            (os.path.join(self.args.hd_data_path,'HD544p_GT/Sintel_Alley2_1280x544.yuv'), 544, 1280),
            (os.path.join(self.args.hd_data_path,'HD544p_GT/Sintel_Market5_1280x544.yuv'), 544, 1280),
            (os.path.join(self.args.hd_data_path,'HD544p_GT/Sintel_Temple1_1280x544.yuv'), 544, 1280),
            (os.path.join(self.args.hd_data_path,'HD544p_GT/Sintel_Temple2_1280x544.yuv'), 544, 1280),
        ]


        if(self.multiple>2):
        	self.framesPath = self.framesPath[:3]
        print("HD Test with multiple ",multiple)
        #self.framesPath.append([npyFirst,npySecond,npyReference])


        self.num_scene = len(self.framesPath)*(100//self.multiple *(self.multiple-1))  # total test scenes
        if len(self.framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.xiph_data_path + "\n"))
        else:
            print("# of HD triplet testset : ", self.num_scene)

    def __getitem__(self, idx):
    	divisor = (100//self.multiple *(self.multiple-1))
    	multiple = self.multiple
    	video_id = self.framesPath[idx//divisor]

    	name = video_id[0]
    	h = video_id[1]
    	w = video_id[2]
    	Reader = YUV_Read(name,h,w,toRGB=True)

    	offset = 1+(idx%divisor)%(multiple-1)
    	i0index = (((idx%divisor)//(multiple-1))*multiple)
    	I0, success0 = Reader.read(i0index)
    	IT,_ = Reader.read(i0index +offset)
    	I1, success1 = Reader.read(i0index+ multiple)


    	#print("I0 ",i0index," | IT",i0index +1+(idx%divisor)%(multiple-1)," | I1",i0index+ multiple)
    	if(not isinstance(I1,np.ndarray)):
    		print("index none: ",idx)
    		return -1,-1,-1,-1
    	if(not isinstance(I0,np.ndarray)):
    		print("index none: ",idx)
    		return -1,-1,-1,-1
    	if(not isinstance(IT,np.ndarray)):
    		print("index none: ",idx)
    		return -1,-1,-1,-1


    	I0 = torch.from_numpy(np.transpose(I0, (2,0,1)).astype("float32") / 255.).cuda()#.unsqueeze(0)#.unsqueeze(2)
    	I1 = torch.from_numpy(np.transpose(I1, (2,0,1)).astype("float32") / 255.).cuda()#.unsqueeze(0)#.unsqueeze(2)
    	IT = torch.from_numpy(np.transpose(IT, (2,0,1)).astype("float32") / 255.).cuda()#.unsqueeze(0)#.unsqueeze(2)
    	I0 = (I0*2)-1
    	I1 = (I1*2)-1
    	IT = (IT*2)-1
    	# if(self.args.HDupscale):
    	# 	I0= F.interpolate(I0.unsqueeze(0),size=(2160,3840),mode="nearest").squeeze(0)
    	# 	I1= F.interpolate(I1.unsqueeze(0),size=(2160,3840),mode="nearest").squeeze(0)
    	# 	IT= F.interpolate(IT.unsqueeze(0),size=(2160,3840),mode="nearest").squeeze(0)
    		#IT0= F.interpolate(IT0.unsqueeze(0),size=(1080,1920),mode="nearest").squeeze(0)
    		#print("diff: ",(IT - IT0).abs().sum())
    	frames = torch.stack([I0,I1,IT],dim=1)
    	#print("I0 ",i0index," | IT",i0index +1+(idx%divisor)%(multiple-1)," ",offset/multiple," | I1",i0index+ multiple)
    	return frames, np.expand_dims(np.array(offset/(multiple), dtype=np.float32), 0), name, ["1", "0", "2"]

    def __len__(self):
        return self.num_scene

class Xiph_Test(data.Dataset):
    def __init__(self, args, validation,twoKC=False):
        self.args = args
        self.framesPath = []
        self.twoKC = twoKC
        assert not validation 
        counter = -1
        for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
            for intFrame in range(2, 99, 2):
                if(validation):
                    counter += 1
                    if(counter % 19 != 0):
                        continue
                npyFirst = os.path.join(args.xiph_data_path, strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
                npySecond = os.path.join(args.xiph_data_path,  strFile + '-' + str(intFrame + 1).zfill(3) + '.png')
                npyReference = os.path.join(args.xiph_data_path,  strFile + '-' + str(intFrame).zfill(3) + '.png')
                self.framesPath.append([npyFirst,npySecond,npyReference])


        self.num_scene = len(self.framesPath)  # total test scenes
        if len(self.framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.xiph_data_path + "\n"))
        else:
            print("# of Xiph triplet testset : ", self.num_scene)

    def __getitem__(self, idx):
        scene_name = self.framesPath[idx][0].split(os.sep)
        scene_name = os.path.join(scene_name[-3], scene_name[-2],scene_name[-1].split(".png")[0])
        I0, I1, It = self.framesPath[idx]
        I0I1It_Path = [I0, I1, It]
        frames = frames_loader_test(self.args, I0I1It_Path, validation=False,Xiph=True)
        
        #frames = RGBframes_np2Tensor(frames, args.img_ch)
        # npyFirst = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame - 1).zfill(3) + '.png', flags=-1)
        #         npySecond = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame + 1).zfill(3) + '.png', flags=-1)
        #         npyReference = cv2.imread(filename='./netflix/' + strFile + '-' + str(intFrame).zfill(3) + '.png', flags=-1)

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]
        #for i in frames:
        #    print("frame: ",i.shape)
        if(self.twoKC):
            frames = frames[:,:,540:-540,1024:-1024]
            assert list(frames.shape) == [3,3,1080,2048]
        if(self.args.xiph2k):
            frames = F.interpolate(frames,scale_factor=1/2,mode="bilinear",align_corners=self.args.align_cornerse)
        return frames, np.expand_dims(np.array(0.5, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.num_scene




def frames_loader_test(args, I0I1It_Path, validation,Xiph=False):
    frames = []
    for path in I0I1It_Path:
        #print(path)
        if(Xiph):
            frame = cv2.imread(path,flags=-1)
        else:
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
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def RGBframes_np2Tensor(imgIn, channel):
    ## input : T, H, W, C
    if channel == 1:
        # rgb --> Y (gray)
        imgIn = np.sum(imgIn * np.reshape([65.481, 128.553, 24.966], [1, 1, 1, 3]) / 255.0, axis=3,
                       keepdims=True) + 16.0

    # to Tensor
    ts = (channel, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [-1,1]
    imgIn = (imgIn / 255.0 - 0.5) * 2

    return imgIn


def make_2D_dataset_X_Train(dir):
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

def make_2D_dataset_Inter4K_Train(dir):
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


class Inter4K_Train(data.Dataset):
    def __init__(self, args,max_t_step_size=16):
        self.args = args
        self.multiple = 8
        self.framesPath = []
        self.max_t_step_size = max_t_step_size
        path = "inter4K/Inter4KFrames/"
        folders = sorted(glob.glob((os.path.join(path,"*"))),key=lambda x: int(x.split("/")[-1]))

                  
        skipit = False
        for i in folders:
            tempath = i#os.path.join(path,i)
            #temfiles = sorted(os.listdir(tempath),key=lambda x: int(x[3:-4]))
            
            for k in os.listdir(tempath):
                #print(k,tempath)
                if(9<int(k.split("_")[1])):
                    #print(tempath)
                    skipit = True
                    break

            if(skipit):
                skipit = False
                continue

            temfiles = sorted(os.listdir(tempath),key=lambda x: int(x[3:-2]))

            scenes = []
            lastscene = 0
            for index,k in enumerate(temfiles):
                if(int(k[-1])>lastscene):
                    scenes.append(index)
                
                lastscene = int(k[-1])
            scenes.append(len(temfiles))

            sceneimages = []
            lastscene = 0
            for scene in scenes:
                temtemfiles = temfiles[lastscene:scene]
                temtemfiles = [os.path.join(tempath,file) for file in temtemfiles]
                sceneimages.append(temtemfiles)
                lastscene = scene
            self.framesPath.append(sceneimages)



        print("# of Inter4K Trainingset : ",len(self.framesPath))
        self.nIterations = len(self.framesPath)

        # Raise error if no images found in test_data_path.
        if len(self.framesPath) == 0:
            if validation:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.val_data_path + "\n"))
            else:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.test_data_path + "\n"))

    def __getitem__(self, idx):
        #start_time = time.time()
        paths = self.framesPath[idx]
        n_scenes = len(paths)
        scenelengths = [len(i) for i in paths]
        t_step_size = 0
        while(t_step_size<3):
            sc_tak = random.randint(0,n_scenes-1)
            t_step_size = min(self.max_t_step_size,(scenelengths[sc_tak]-1)//2)


        t_step_size = random.randint(2, t_step_size)
        t_list = np.linspace((1 / t_step_size), (1 - (1 / t_step_size)), (t_step_size - 1))
        candidate_frames = paths[sc_tak]
        firstFrameIdx = random.randint(0, (scenelengths[sc_tak] - t_step_size)-1)
        interIdx = random.randint(1, t_step_size - 1)  # relative index, 1~self.t_step_size-1
        interFrameIdx = firstFrameIdx + interIdx  # absolute index
        t_value = t_list[interIdx - 1]  # [0,1]
        #print([firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx],t_value)
        ########    TEMPORAL DATA AUGMENTATION      ####################
        if (random.randint(0, 1)):
            frameRange = [firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + t_step_size, firstFrameIdx, interFrameIdx]
            interIdx = t_step_size - interIdx  # (self.t_step_size-1) ~ 1
            t_value = 1.0 - t_value


        ##########      SPATIAL DATA AUGMENTATION       ################################
        frames = frames_loader_train_inter4k(self.args, candidate_frames,
                                        frameRange)  # including "np2Tensor [-1,1] normalized"

        #print("time needed: ",time.time()-start_time)
        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nIterations

def frames_loader_train_inter4k(args, candidate_frames, frameRange):
    #start_time = time.time()
    frames = []
    if(args.patch_size>512):
        res_choices = ["im2k.png","im4k.png"]
    else:
        res_choices = ["im1k.png","im2k.png","im4k.png"]
    ranval = random.randint(0,len(res_choices)-1)
    res = res_choices[ranval]
    for frameIndex in frameRange:
        frame =np.array(Image.open(os.path.join(candidate_frames[frameIndex],res)))
        #print("loaded: ",os.path.join(candidate_frames[frameIndex],res))
        # print(frame.shape)
        # frame = cv2.imread(candidate_frames[frameIndex])
        frames.append(frame)

    #print("Time needed image read:",time.time()-start_time)

    # [2160,3840] from cv2.imread!
    # [3840,2160]



    (ih, iw, c) = frames[0].shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if args.need_patch:  ## random crop
            ps = args.patch_size
            ix = random.randrange(0, iw - ps + 1)
            iy = random.randrange(0, ih - ps + 1)
            frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    if random.random() < 0.5:  # random horizontal flip
        frames = frames[:, :, ::-1, :]

    # No vertical flip

    rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames


def frames_loader_train(args, candidate_frames, frameRange,inter4k=False):
    #start_time = time.time()
    frames = []
    for frameIndex in frameRange:
        frame =np.array(Image.open(candidate_frames[frameIndex]))
        # print(frame.shape)
        # frame = cv2.imread(candidate_frames[frameIndex])
        frames.append(frame)

    #print("Time needed image read:",time.time()-start_time)

    # [2160,3840] from cv2.imread!
    # [3840,2160]
    #start_time = time.time()
    if inter4k:
        res_choices = [(3840,2160),(1920,1080),(960,540)]
        ranval = random.randint(0,len(res_choices)-1)
        res = res_choices[ranval]
        if(ranval != 0):
            for index,img in enumerate(frames):
                frames[index] = cv2.resize(frames[index],dsize=res,interpolation=cv2.INTER_AREA)
        else:
            for index,img in enumerate(frames):
                frames[index] = frames[index][540:-540,960:-960,:]

    #print("time needed calcs: ",time.time()-start_time)



    (ih, iw, c) = frames[0].shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)
    if args.need_patch:  ## random crop
            ps = args.patch_size
            ix = random.randrange(0, iw - ps + 1)
            iy = random.randrange(0, ih - ps + 1)
            frames = frames[:, iy:iy + ps, ix:ix + ps, :]

    if random.random() < 0.5:  # random horizontal flip
        frames = frames[:, :, ::-1, :]

    # No vertical flip

    rot = random.randint(0, 3)  # random rotate
    frames = np.rot90(frames, rot, (1, 2))

    """ np2Tensor [-1,1] normalized """
    frames = RGBframes_np2Tensor(frames, args.img_ch)

    return frames

class X_Train(data.Dataset):
    def __init__(self, args, max_t_step_size,device,deter=False):
        self.args = args
        self.deter = deter
        self.max_t_step_size = max_t_step_size
        self.device = device
        self.framesPath = make_2D_dataset_X_Train(self.args.train_data_path)
        self.nScenes = len(self.framesPath)

        self.psnr_bil =  AverageClass('PSNR bil:', ':.4e')
        self.psnr_dct = AverageClass('PSNR dct:', ':.4e')
        # Raise error if no images found in train_data_path.
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.train_data_path + "\n"))

    def __getitem__(self, idx):
        if(self.deter):
            random.seed(10)
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
        frames = frames_loader_train(self.args, candidate_frames,
                                     frameRange)  # including "np2Tensor [-1,1] normalized"




        #frames: torch.Size([3, 3, 384, 384]) 
        downscaling = 8
        if(self.args.give_dct_ds_in):
            wiS = 20
            components =  int((wiS*wiS)/(downscaling*downscaling)) 
            for ind in range(2):
                temp_pic = (frames[:,ind,:,:].numpy() )
                other_pic = np.copy(temp_pic)
                frames[:,ind,:,:] = (torch.from_numpy(pca_trans_and_back((temp_pic+ 1)/2,components,wiS))*2)-1
                #print("Error: ",np.sum(np.abs(frames[:,ind,:,:].numpy()  - other_pic)))
                #print("PSNR: ", psnr(other_pic,frames[:,ind,:,:].numpy()))
                self.psnr_dct.update(psnr(other_pic,frames[:,ind,:,:].numpy()),1)
        if(self.args.give_bilin_in):
            for ind in range(2):
                tem_sh = frames.shape
                temp_pic = frames[:,ind,:,:].clone()
                temp = cv2.resize(frames[:,ind,:,:].numpy().reshape(tem_sh[2],tem_sh[3],tem_sh[1]) ,(tem_sh[3]//downscaling,tem_sh[2]//downscaling))
                frames[:,ind,:,:] = torch.from_numpy( cv2.resize(temp,(tem_sh[3],tem_sh[2])).reshape(tem_sh[0],tem_sh[2],tem_sh[3]) )
                #print("PSNR: ", psnr(temp_pic.numpy(),frames[:,ind,:,:].numpy()))
                #print("Error: ",torch.sum(torch.abs(temp_pic - frames[:,ind,:,:])).item())
                self.psnr_bil.update(psnr(temp_pic.numpy(),frames[:,ind,:,:].numpy()),1)

        if(self.deter):
            random.seed(self.args.exp_num)
        if(self.args.samefirstpic):
            random.seed(100)
        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes


def toTenD(arr,device):
    return torch.from_numpy(arr).to(device)

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

class Inter4K_TestExtreme(data.Dataset):
    def __init__(self, args, multiple):
        self.args = args
        self.multiple = 8
        self.testPath = []
        # Return """ make [I0,I1,It,t,scene_folder] """
        path = "inter4K/Inter4KFrames/"
        folders = os.listdir(path)
        t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
        for i in folders:
            tempath = os.path.join(path,i)
            temfiles = sorted(os.listdir(tempath),key=lambda x: int(x[3:-4]))

            #temrange = min(len(temfiles)//(multiple+1),2)
            #for k in range(temrange):
            temtemfiles = temfiles[:33]
            temtemfiles = [os.path.join(tempath,file) for file in temtemfiles]
            for ind in range(multiple-1):
                self.testPath.append([temtemfiles[0],temtemfiles[-1],temtemfiles[(ind+1)*(4)],t[ind],i])
            
        #print(self.testPath)
        print("# of Inter4K triplet testset : ",len(self.testPath))
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
        frames = frames_loader_test(self.args, I0I1It_Path,validation=False)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        #torch_prints(frames,"frames ")

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations



# folders = sorted(glob.glob((os.path.join(path,"*"))),key=lambda x: int(x.split("/")[-1]))

                  
#         skipit = False
#         for i in folders:
#             tempath = i#os.path.join(path,i)
#             #temfiles = sorted(os.listdir(tempath),key=lambda x: int(x[3:-4]))
            
#             for k in os.listdir(tempath):
#                 #print(k,tempath)
#                 if(9<int(k.split("_")[1])):
#                     #print(tempath)
#                     skipit = True
#                     break

#             if(skipit):
#                 skipit = False
#                 continue

#             temfiles = sorted(os.listdir(tempath),key=lambda x: int(x[3:-2]))

#             scenes = []
#             lastscene = 0
#             for index,k in enumerate(temfiles):
#                 if(int(k[-1])>lastscene):
#                     scenes.append(index)
                
#                 lastscene = int(k[-1])
#             scenes.append(len(temfiles))

#             sceneimages = []
#             lastscene = 0
#             for scene in scenes:
#                 temtemfiles = temfiles[lastscene:scene]
#                 temtemfiles = [os.path.join(tempath,file) for file in temtemfiles]
#                 sceneimages.append(temtemfiles)
#                 lastscene = scene
#             self.framesPath.append(sceneimages)
class Inter4K_Test(data.Dataset):
    def __init__(self, args, multiple,twoK=False,scenerange=8):
        self.args = args
        self.twoK = False
        self.multiple = 8
        self.scenerange = scenerange
        assert self.scenerange%self.multiple == 0
        self.testPath = []
        # Return """ make [I0,I1,It,t,scene_folder] """
        testfolder = "inter4K/Inter4KNewTestset/"
        folders = os.listdir(testfolder)
        t = np.linspace((1 / self.multiple), (1 - (1 / self.multiple)), (self.multiple - 1))
        
        skipit = False
        for i in folders:
            tempath = os.path.join(testfolder,i)
            

            temfiles = sorted(os.listdir(tempath),key=lambda x: int(x.split("_")[0][3:]))
            scenes = [[]]
            lastscene = 0
            for imadd in temfiles:
                if(int(imadd.split("_")[1][:-4])> lastscene):
                    scenes.append([imadd])
                    lastscene += 1
                else:
                    scenes[-1].append(imadd)

            
            # if(len(scenes)<4):
            #     continue
            # print(scenes)

            for scenindex,scen in enumerate(scenes):
                if(not len(scen)< self.scenerange+1):
                    temscen = [os.path.join(tempath,file) for file in scen]
                    for temk in range(self.multiple-1):
                        self.testPath.append([temscen[0],temscen[self.scenerange],temscen[(temk+1)*(self.scenerange//self.multiple)],t[temk],tempath+"_scene_"+str(scenindex)])

            # for temk in self.testPath:
            #     print(temk)
            #     print()
            # #print(self.testPath)
            # sys.exit()
            

            
        #print(self.testPath)
        print("# of Inter4K triplet testset : ",len(self.testPath))
        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.test_data_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]

        # Returns all frames (65)
        frames = frames_loader_test(self.args, I0I1It_Path,validation=False)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        #torch_prints(frames,"frames ")
        # if(self.twoKC):
        #     frames = frames[:,:,540:-540,1024:-1024]
        #     assert list(frames.shape) == [3,3,1080,2048]
        if(self.twoK):
            frames = F.interpolate(frames,scale_factor=1/2,mode="bilinear",align_corners=self.args.align_cornerse)
        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations

class X_Test(data.Dataset):
    def __init__(self, args, multiple, validation,twoKC=False):
        self.args = args
        self.multiple = multiple
        self.validation = validation
        self.twoKC = twoKC
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

        # downscaling = 8
        # if(self.args.give_dct_ds_in):
        #     wiS = 20
        #     components =  int((wiS*wiS)/(downscaling*downscaling)) 
        #     for ind in range(2):
        #         temp_pic = (frames[:,ind,:,:].numpy() )
        #         other_pic = np.copy(temp_pic)
        #         frames[:,ind,:,:] = (torch.from_numpy(pca_trans_and_back((temp_pic+ 1)/2,components,wiS))*2)-1
        #         #print("Error: ",np.sum(np.abs(frames[:,ind,:,:].numpy()  - other_pic)))
        #         #print("PSNR: ", psnr(other_pic,frames[:,ind,:,:].numpy()))
        #         self.psnr_dct.update(psnr(other_pic,frames[:,ind,:,:].numpy()),1)
        # if(self.args.give_bilin_in):
        #     for ind in range(2):
        #         tem_sh = frames.shape
        #         temp_pic = frames[:,ind,:,:].clone()
        #         temp = cv2.resize(frames[:,ind,:,:].numpy().reshape(tem_sh[2],tem_sh[3],tem_sh[1]) ,(tem_sh[3]//downscaling,tem_sh[2]//downscaling))
        #         frames[:,ind,:,:] = torch.from_numpy( cv2.resize(temp,(tem_sh[3],tem_sh[2])).reshape(tem_sh[0],tem_sh[2],tem_sh[3]) )
        #         #print("PSNR: ", psnr(temp_pic.numpy(),frames[:,ind,:,:].numpy()))
        #         #print("Error: ",torch.sum(torch.abs(temp_pic - frames[:,ind,:,:])).item())
        #         self.psnr_bil.update(psnr(temp_pic.numpy(),frames[:,ind,:,:].numpy()),1)
        #print("X_test frames shape: ", frames.shape)
        

        if(self.args.xtest2k):
            frames = F.interpolate(frames,scale_factor=1/2,mode="bilinear",align_corners=self.args.align_cornerse)
        if(self.twoKC):
            frames = frames[:,:,540:-540,1024:-1024]
        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


class Vimeo_Train(data.Dataset):
    def __init__(self, args):
        self.args = args
        self.t = 0.5
        self.framesPath = []
        f = open(os.path.join(args.vimeo_data_path, 'tri_trainlist.txt'),
                 'r')  # '../Datasets/vimeo_triplet/sequences/tri_trainlist.txt'
        while True:
            scene_path = f.readline().split('\n')[0]
            if not scene_path: break
            frames_list = sorted(glob.glob(os.path.join(args.vimeo_data_path, 'sequences', scene_path,
                                                        '*.png')))  # '../Datasets/vimeo_triplet/sequences/%05d/%04d/*.png'
            self.framesPath.append(frames_list)
        f.close
        # self.framesPath = self.framesPath[:20]
        self.nScenes = len(self.framesPath)
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.vimeo_data_path + "\n"))
        print("nScenes of Vimeo train triplet : ", self.nScenes)

    def __getitem__(self, idx):
        candidate_frames = self.framesPath[idx]

        """ Randomly reverse frames """
        if (random.randint(0, 1)):
            frameRange = [0, 2, 1]
        else:
            frameRange = [2, 0, 1]
        frames = frames_loader_train(self.args, candidate_frames,
                                     frameRange)  # including "np2Tensor [-1,1] normalized"

        #print("bin hier",frames.shape,np.expand_dims(np.array(0.5, dtype=np.float32), 0))
        return frames, np.expand_dims(np.array(0.5, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes


class Vimeo_Test(data.Dataset):
    def __init__(self, args, validation):
        self.args = args
        self.framesPath = []
        f = open(os.path.join(args.vimeo_data_path, 'tri_testlist.txt'), 'r')
        while True:
            scene_path = f.readline().split('\n')[0]
            if not scene_path: break
            frames_list = sorted(glob.glob(os.path.join(args.vimeo_data_path, 'sequences', scene_path,
                                                        '*.png')))  # '../Datasets/vimeo_triplet/sequences/%05d/%04d/*.png'
            self.framesPath.append(frames_list)
        if validation:
            self.framesPath = self.framesPath[::37]
        f.close

        self.num_scene = len(self.framesPath)  # total test scenes
        if len(self.framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + args.vimeo_data_path + "\n"))
        else:
            print("# of Vimeo triplet testset : ", self.num_scene)

    def __getitem__(self, idx):
        scene_name = self.framesPath[idx][0].split(os.sep)
        scene_name = os.path.join(scene_name[-3], scene_name[-2])
        I0, It, I1 = self.framesPath[idx]
        I0I1It_Path = [I0, I1, It]
        frames = frames_loader_test(self.args, I0I1It_Path, validation=False)

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]
        print(frames[0].shape)
        return frames, np.expand_dims(np.array(0.5, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.num_scene

def make_2D_dataset_Custom_Test(dir, multiple):
    """ make [I0,I1,It,t,scene_folder] """
    """ 1D (accumulated) """
    testPath = []
    t = np.linspace((1 / multiple), (1 - (1 / multiple)), (multiple - 1))
    for scene_folder in sorted(glob.glob(os.path.join(dir, '*', ''))):  # [scene1, scene2, scene3, ...]
        frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # ex) ['00000.png',...,'00123.png']
        for idx in range(0, len(frame_folder)):
            if idx == len(frame_folder) - 1:
                break
            for suffix, mul in enumerate(range(multiple - 1)):
                I0I1It_paths = []
                I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
                I0I1It_paths.append(frame_folder[idx + 1])  # I1 (fix)
                target_t_Idx = frame_folder[idx].split(os.sep)[-1].split('.')[0]+'_' + str(suffix).zfill(3) + '.png'
                # ex) target t name: 00017.png => '00017_1.png'
                I0I1It_paths.append(os.path.join(scene_folder, target_t_Idx))  # It
                I0I1It_paths.append(t[mul]) # t
                I0I1It_paths.append(frame_folder[idx].split(os.path.join(dir, ''))[-1].split(os.sep)[0])  # scene1
                testPath.append(I0I1It_paths)
    return testPath


# def make_2D_dataset_Custom_Test(dir):
#     """ make [I0,I1,It,t,scene_folder] """
#     """ 1D (accumulated) """
#     testPath = []
#     for scene_folder in sorted(glob.glob(os.path.join(dir, '*/'))):  # [scene1, scene2, scene3, ...]
#         frame_folder = sorted(glob.glob(scene_folder + '*.png'))  # ex) ['00000.png',...,'00123.png']
#         for idx in range(0, len(frame_folder)):
#             if idx == len(frame_folder) - 1:
#                 break
#             I0I1It_paths = []
#             I0I1It_paths.append(frame_folder[idx])  # I0 (fix)
#             I0I1It_paths.append(frame_folder[idx + 1])  # I1 (fix)
#             target_t_Idx = frame_folder[idx].split('/')[-1].split('.')[0]+'_x2.png'
#             # ex) target t name: 00017.png => '00017_1.png'
#             I0I1It_paths.append(os.path.join(scene_folder, target_t_Idx))  # It
#             I0I1It_paths.append(0.5) # t
#             I0I1It_paths.append(frame_folder[idx].split(os.path.join(dir, ''))[-1].split('/')[0])  # scene1
#             testPath.append(I0I1It_paths)
#     for asdf in testPath:
#         print(asdf)
#     return testPath


class Custom_Test(data.Dataset):
    def __init__(self, args, multiple):
        self.args = args
        self.multiple = multiple
        self.testPath = make_2D_dataset_Custom_Test(self.args.custom_path, self.multiple)
        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.custom_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]
        dummy_dir = I1 # due to there is not ground truth intermediate frame.
        I0I1It_Path = [I0, I1, dummy_dir]

        frames = frames_loader_test(self.args, I0I1It_Path, None)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.epsilon = 1e-3

    def forward(self, X, Y):
        loss = torch.mean(torch.sqrt((X - Y) ** 2 + self.epsilon ** 2))
        return loss


def set_rec_loss(args):
    loss_type = args.loss_type
    if loss_type == 'MSE':
        lossfunction = nn.MSELoss()
    elif loss_type == 'L1':
        lossfunction = nn.L1Loss()
    elif loss_type == 'L1_Charbonnier_loss':
        lossfunction = L1_Charbonnier_loss()

    return lossfunction


class AverageClass(object):
    """ For convenience of averaging values """
    """ refer from "https://github.com/pytorch/examples/blob/master/imagenet/main.py" """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} (avg:{avg' + self.fmt + '})'
        # Accm_Time[s]: 1263.517 (avg:639.701)    (<== if AverageClass('Accm_Time[s]:', ':6.3f'))
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    """ For convenience of printing diverse values by using "AverageClass" """
    """ refer from "https://github.com/pytorch/examples/blob/master/imagenet/main.py" """

    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        # # Epoch: [0][  0/196]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def metrics_evaluation_X_Test(pred_save_path, test_data_path, metrics_types, flow_flag=False, multiple=8, server=None):
    """
        pred_save_path = './test_img_dir/XVFInet_exp1/epoch_00099' when 'args.epochs=100'
        test_data_path = ex) 'F:/Jihyong/4K_1000fps_dataset/VIC_4K_1000FPS/X_TEST'
            format: -type1
                        -scene1
                            :
                        -scene5
                    -type2
                            :
                    -type3
                        :
                        -scene5
        "metrics_types": ["PSNR", "SSIM", "LPIPS", "tOF", "tLP100"]
        "flow_flag": option for saving motion visualization
        "final_test_type": ['first_interval', 1, 2, 3, 4]
        "multiple": x4, x8, x16, x32 for interpolation
     """

    pred_framesPath = []
    for type_folder in sorted(glob.glob(os.path.join(pred_save_path, '*', ''))):  # [type1,type2,type3,...]
        for scene_folder in sorted(glob.glob(os.path.join(type_folder, '*', ''))):  # [scene1,scene2,..]
            scene_framesPath = []
            for frame_path in sorted(glob.glob(scene_folder + '*.png')):
                scene_framesPath.append(frame_path)
            pred_framesPath.append(scene_framesPath)
    if len(pred_framesPath) == 0:
        raise (RuntimeError("Found 0 files in " + pred_save_path + "\n"))

    # GT_framesPath = make_2D_dataset_X_Test(test_data_path, multiple, t_step_size=32)
    # pred_framesPath = make_2D_dataset_X_Test(pred_save_path, multiple, t_step_size=32)

    # ex) pred_save_path: './test_img_dir/XVFInet_exp1/epoch_00099' when 'args.epochs=100'
    # ex) framesPath: [['./VIC_4K_1000FPS/VIC_Test/Fast/003_TEST_Fast/00000.png',...], ..., []] 2D List, len=30
    # ex) scenesFolder: ['Fast/003_TEST_Fast',...]

    keys = metrics_types
    len_dict = dict.fromkeys(keys, 0)
    Total_avg_dict = dict.fromkeys(["TotalAvg_" + _ for _ in keys], 0)
    Type1_dict = dict.fromkeys(["Type1Avg_" + _ for _ in keys], 0)
    Type2_dict = dict.fromkeys(["Type2Avg_" + _ for _ in keys], 0)
    Type3_dict = dict.fromkeys(["Type3Avg_" + _ for _ in keys], 0)

    # LPIPSnet = dm.DistModel()
    # LPIPSnet.initialize(model='net-lin', net='alex', use_gpu=True)

    total_list_dict = {}
    key_str = 'Metrics -->'
    for key_i in keys:
        total_list_dict[key_i] = []
        key_str += ' ' + str(key_i)
    key_str += ' will be measured.'
    print(key_str)

    for scene_idx, scene_folder in enumerate(pred_framesPath):
        per_scene_list_dict = {}
        for key_i in keys:
            per_scene_list_dict[key_i] = []
        pred_candidate = pred_framesPath[scene_idx]  # get all frames in pred_framesPath
        # GT_candidate = GT_framesPath[scene_idx]  # get 4800 frames
        # num_pred_frame_per_folder = len(pred_candidate)

        # save_path = os.path.join(pred_save_path, pred_scenesFolder[scene_idx])
        save_path = scene_folder[0]
        # './test_img_dir/XVFInet_exp1/epoch_00099/type1/scene1'

        # excluding both frame0 and frame1 (multiple of 32 indices)
        for frameIndex, pred_frame in enumerate(pred_candidate):
            # if server==87:
            # GTinterFrameIdx = pred_frame.split('/')[-1]  # ex) 8, when multiple = 4, # 87 server
            # else:
            # GTinterFrameIdx = pred_frame.split('\\')[-1]  # ex) 8, when multiple = 4
            # if not (GTinterFrameIdx % 32) == 0:
            if frameIndex > 0 and frameIndex < multiple:
                """ only compute predicted frames (excluding multiples of 32 indices), ex) 8, 16, 24, 40, 48, 56, ... """
                output_img = cv2.imread(pred_frame).astype(np.float32)  # BGR, [0,255]
                target_img = cv2.imread(pred_frame.replace(pred_save_path, test_data_path)).astype(
                    np.float32)  # BGR, [0,255]
                pred_frame_split = pred_frame.split(os.sep)
                msg = "[x%d] frame %s, " % (
                multiple, os.path.join(pred_frame_split[-3], pred_frame_split[-2], pred_frame_split[-1]))  # per frame

                if "tOF" in keys:  # tOF
                    # if (GTinterFrameIdx % 32) == int(32/multiple):
                    # if (frameIndex % multiple) == 1:
                    if frameIndex == 1:
                        # when first predicted frame in each interval
                        pre_out_grey = cv2.cvtColor(cv2.imread(pred_candidate[0]).astype(np.float32),
                                                    cv2.COLOR_BGR2GRAY)  #### CAUTION BRG
                        # pre_tar_grey = cv2.cvtColor(cv2.imread(pred_candidate[0].replace(pred_save_path, test_data_path)), cv2.COLOR_BGR2GRAY)  #### CAUTION BRG
                        pre_tar_grey = pre_out_grey  #### CAUTION BRG

                    # if not H_match_flag or not W_match_flag:
                    #    pre_tar_grey = pre_tar_grey[:new_t_H, :new_t_W, :]

                    # pre_tar_grey = pre_out_grey

                    output_grey = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
                    target_grey = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)

                    target_OF = cv2.calcOpticalFlowFarneback(pre_tar_grey, target_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    output_OF = cv2.calcOpticalFlowFarneback(pre_out_grey, output_grey, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # target_OF, ofy, ofx = crop_8x8(target_OF) #check for size reason
                    # output_OF, ofy, ofx = crop_8x8(output_OF)
                    OF_diff = np.absolute(target_OF - output_OF)
                    if flow_flag:
                        """ motion visualization """
                        flow_path = save_path + '_tOF_flow'
                        check_folder(flow_path)
                        # './test_img_dir/XVFInet_exp1/epoch_00099/Fast/003_TEST_Fast_tOF_flow'
                        tOFpath = os.path.join(flow_path, "tOF_flow_%05d.png" % (GTinterFrameIdx))
                        # ex) "./test_img_dir/epoch_005/Fast/003_TEST_Fast/00008_tOF" when start_idx=0, multiple=4, frameIndex=0
                        hsv = np.zeros_like(output_img)  # check for size reason
                        hsv[..., 1] = 255
                        mag, ang = cv2.cartToPolar(OF_diff[..., 0], OF_diff[..., 1])
                        # print("tar max %02.6f, min %02.6f, avg %02.6f" % (mag.max(), mag.min(), mag.mean()))
                        maxV = 0.4
                        mag = np.clip(mag, 0.0, maxV) / maxV
                        hsv[..., 0] = ang * 180 / np.pi / 2
                        hsv[..., 2] = mag * 255.0  #
                        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        cv2.imwrite(tOFpath, bgr)
                        print("png for motion visualization has been saved in [%s]" %
                              (flow_path))
                    OF_diff_tmp = np.sqrt(np.sum(OF_diff * OF_diff, axis=-1)).mean()  # l1 vector norm
                    # OF_diff, ofy, ofx = crop_8x8(OF_diff)
                    total_list_dict["tOF"].append(OF_diff_tmp)
                    per_scene_list_dict["tOF"].append(OF_diff_tmp)
                    msg += "tOF %02.2f, " % (total_list_dict["tOF"][-1])

                    pre_out_grey = output_grey
                    pre_tar_grey = target_grey

                # target_img, ofy, ofx = crop_8x8(target_img)
                # output_img, ofy, ofx = crop_8x8(output_img)

                if "PSNR" in keys:  # psnr
                    psnr_tmp = psnr(target_img, output_img)
                    total_list_dict["PSNR"].append(psnr_tmp)
                    per_scene_list_dict["PSNR"].append(psnr_tmp)
                    msg += "PSNR %02.2f" % (total_list_dict["PSNR"][-1])

                if "SSIM" in keys:  # ssim
                    ssim_tmp = ssim_bgr(target_img, output_img)
                    total_list_dict["SSIM"].append(ssim_tmp)
                    per_scene_list_dict["SSIM"].append(ssim_tmp)

                    msg += ", SSIM %02.2f" % (total_list_dict["SSIM"][-1])

                # msg += ", crop (%d, %d)" % (ofy, ofx) # per frame (not scene)
                print(msg)

        """ after finishing one scene """
        per_scene_pd_dict = {}  # per scene
        for cur_key in keys:
            # save_path = './test_img_dir/XVFInet_exp1/epoch_00099/Fast/003_TEST_Fast'
            num_data = cur_key + "_[x%d]_[%s]" % (multiple, save_path.split(os.sep)[-2])  # '003_TEST_Fast'
            # num_data => ex) PSNR_[x8]_[041_TEST_Fast]
            """ per scene """
            per_scene_cur_list = np.float32(per_scene_list_dict[cur_key])
            per_scene_pd_dict[num_data] = pd.Series(per_scene_cur_list)  # dictionary
            per_scene_num_data_sum = per_scene_cur_list.sum()
            per_scene_num_data_len = per_scene_cur_list.shape[0]
            per_scene_num_data_mean = per_scene_num_data_sum / per_scene_num_data_len
            """ accumulation """
            cur_list = np.float32(total_list_dict[cur_key])
            num_data_sum = cur_list.sum()
            num_data_len = cur_list.shape[0]  # accum
            num_data_mean = num_data_sum / num_data_len
            print(" %s, (per scene) max %02.4f, min %02.4f, avg %02.4f" %
                  (num_data, per_scene_cur_list.max(), per_scene_cur_list.min(), per_scene_num_data_mean))  #

            Total_avg_dict["TotalAvg_" + cur_key] = num_data_mean  # accum, update every iteration.

            len_dict[cur_key] = num_data_len  # accum, update every iteration.

            # folder_dict["FolderAvg_" + cur_key] += num_data_mean
            if scene_idx < 5:
                Type1_dict["Type1Avg_" + cur_key] += per_scene_num_data_mean
            elif (scene_idx >= 5) and (scene_idx < 10):
                Type2_dict["Type2Avg_" + cur_key] += per_scene_num_data_mean
            elif (scene_idx >= 10) and (scene_idx < 15):
                Type3_dict["Type3Avg_" + cur_key] += per_scene_num_data_mean

        mode = 'w' if scene_idx == 0 else 'a'

        total_csv_path = os.path.join(pred_save_path, "total_metrics.csv")
        # ex) pred_save_path: './test_img_dir/XVFInet_exp1/epoch_00099' when 'args.epochs=100'
        pd.DataFrame(per_scene_pd_dict).to_csv(total_csv_path, mode=mode)

    """ combining all results after looping all scenes. """
    for key in keys:
        Total_avg_dict["TotalAvg_" + key] = pd.Series(
            np.float32(Total_avg_dict["TotalAvg_" + key]))  # replace key (update)
        Type1_dict["Type1Avg_" + key] = pd.Series(np.float32(Type1_dict["Type1Avg_" + key] / 5))  # replace key (update)
        Type2_dict["Type2Avg_" + key] = pd.Series(np.float32(Type2_dict["Type2Avg_" + key] / 5))  # replace key (update)
        Type3_dict["Type3Avg_" + key] = pd.Series(np.float32(Type3_dict["Type3Avg_" + key] / 5))  # replace key (update)

        print("%s, total frames %d, total avg %02.4f, Type1 avg %02.4f, Type2 avg %02.4f, Type3 avg %02.4f" %
              (key, len_dict[key], Total_avg_dict["TotalAvg_" + key],
               Type1_dict["Type1Avg_" + key], Type2_dict["Type2Avg_" + key], Type3_dict["Type3Avg_" + key]))

    pd.DataFrame(Total_avg_dict).to_csv(total_csv_path, mode='a')
    pd.DataFrame(Type1_dict).to_csv(total_csv_path, mode='a')
    pd.DataFrame(Type2_dict).to_csv(total_csv_path, mode='a')
    pd.DataFrame(Type3_dict).to_csv(total_csv_path, mode='a')

    print("csv file of all metrics for all scenes has been saved in [%s]" %
          (total_csv_path))
    print("Finished.")


def to_uint8(x, vmin, vmax):
    ##### color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    x = x.astype('float32')
    x = (x - vmin) / (vmax - vmin) * 255  # 0~255
    return np.clip(np.round(x), 0, 255)


def psnr(img_true, img_pred,args):
    ##### PSNR with color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    """
    # img format : [h,w,c], RGB
    """
    # Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255), 255)[:, :, 0]
    # Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255), 255)[:, :, 0]
    if(not args.XVFIPSNR):
        return peak_signal_noise_ratio(image_true=img_true,image_test=img_pred,data_range=255)

    # XVFI Code
    diff = img_true - img_pred
    rmse = np.sqrt(np.mean(np.power(diff, 2)))
    if rmse == 0:
        return float('inf')
    return 20 * np.log10(255. / rmse)


def ssim_bgr(img_true, img_pred):  ##### SSIM for BGR, not RGB #####
    """
    # img format : [h,w,c], BGR
    """
    Y_true = _rgb2ycbcr(to_uint8(img_true, 0, 255)[:, :, ::-1], 255)[:, :, 0]
    Y_pred = _rgb2ycbcr(to_uint8(img_pred, 0, 255)[:, :, ::-1], 255)[:, :, 0]
    # return compare_ssim(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())
    return structural_similarity(Y_true, Y_pred, data_range=Y_pred.max() - Y_pred.min())


def im2tensor(image, imtype=np.uint8, cent=1., factor=255. / 2.):
    # def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


# [0,255]2[-1,1]2[1,3,H,W]-shaped

def denorm255(x):
    out = (x + 1.0) / 2.0
    return out.clamp_(0.0, 1.0) * 255.0


def denorm255_np(x):
    # numpy
    out = (x + 1.0) / 2.0
    return out.clip(0.0, 1.0) * 255.0


def _rgb2ycbcr(img, maxVal=255):
    ##### color space transform, originally from https://github.com/yhjo09/VSR-DUF #####
    O = np.array([[16],
                  [128],
                  [128]])
    T = np.array([[0.256788235294118, 0.504129411764706, 0.097905882352941],
                  [-0.148223529411765, -0.290992156862745, 0.439215686274510],
                  [0.439215686274510, -0.367788235294118, -0.071427450980392]])

    if maxVal == 1:
        O = O / 255.0

    t = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
    t = np.dot(t, np.transpose(T))
    t[:, 0] += O[0]
    t[:, 1] += O[1]
    t[:, 2] += O[2]
    ycbcr = np.reshape(t, [img.shape[0], img.shape[1], img.shape[2]])

    return ycbcr

class set_warping_loss_endflow(nn.Module):
    def __init__(self, args):
        super(set_warping_loss_endflow, self).__init__()
        self.args = args
        self.device = self.args.gpu
# Img: [B, C, T, H, W] 
    def forward(self, img,gt, flows):
        loss = 0.0
        first = img[:,:,0,:,:]
        second = img[:,:,1,:,:]
        flow_t0_l = flows[0]
        flow_t1_l = flows[1]
        warped_fir = self.bwarp(first, flow_t0_l)
        warped_sec = self.bwarp(second, flow_t1_l)

        loss_f = nn.L1Loss()
        loss = loss_f(warped_sec,gt) + loss_f(warped_fir,gt)
        return loss
    def bwarp(self, x, flo,withmask=True):
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
        output = nn.functional.grid_sample(x, vgrid)#, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(self.device)
        mask = nn.functional.grid_sample(mask, vgrid)#, align_corners=True)

        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        if(withmask):
            return output * mask
        else:
            return output

class set_warping_loss(nn.Module):
    def __init__(self, args):
        super(set_warping_loss, self).__init__()
        self.args = args
        self.device = self.args.gpu
# Img: [B, C, T, H, W] 
    def forward(self, img, flows):
        loss = 0.0
        if(False):#self.args.pyramid_warp_loss):
            # TODO but not so important i think
            first = img[:,:,0,:,:]
            second = img[:,:,1,:,:]
            flow_01_l = flows[:,:2,:,:]
            flow_10_l = flows[:,2:,:,:]
            warped_sec = self.bwarp(second, flow_01_l)
            warped_fir = self.bwarp(first, flow_10_l)

            loss_f = nn.L1Loss()
            loss = loss_f(warped_sec,first) + loss_f(warped_fir,second)
            return None
        else:
            first = img[:,:,0,:,:]
            second = img[:,:,1,:,:]
            flow_01_l = flows[:,:2,:,:]
            flow_10_l = flows[:,2:,:,:]
            warped_sec = self.bwarp(second, flow_01_l)
            warped_fir = self.bwarp(first, flow_10_l)

            loss_f = nn.L1Loss()
            loss = loss_f(warped_sec,first) + loss_f(warped_fir,second)
            return loss
    def bwarp(self, x, flo,withmask=True):
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
        output = nn.functional.grid_sample(x, vgrid)#, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(self.device)
        mask = nn.functional.grid_sample(mask, vgrid)#, align_corners=True)

        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        if(withmask):
            return output * mask
        else:
            return output
        

class set_smoothness_loss(nn.Module):
    def __init__(self, weight=150.0, edge_aware=True):
        super(set_smoothness_loss, self).__init__()
        self.edge_aware = edge_aware
        self.weight = weight ** 2

    def forward(self, flow, img):
        img_gh = torch.mean(torch.pow((img[:, :, 1:, :] - img[:, :, :-1, :]), 2), dim=1).unsqueeze(1)#, keepdims=True)
        img_gw = torch.mean(torch.pow((img[:, :, :, 1:] - img[:, :, :, :-1]), 2), dim=1).unsqueeze(1)#, keepdims=True)

        weight_gh = torch.exp(-self.weight * img_gh)
        weight_gw = torch.exp(-self.weight * img_gw)

        flow_gh = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        flow_gw = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
        if self.edge_aware:
            return (torch.mean(weight_gh * flow_gh) + torch.mean(weight_gw * flow_gw)) * 0.5
        else:
            return (torch.mean(flow_gh) + torch.mean(flow_gw)) * 0.5


class set_warping_loss_endflow_forward(nn.Module):
    def __init__(self):
        super(set_warping_loss_endflow_forward, self).__init__()
        #self.args = args
        #self.device = self.args.gpu
        self.softsplat = Softsplat()
# Img: [B, C, T, H, W] 
    def forward(self, flows,im0, im1,t_value):
        # 0 -> t :2
        # 1 -> t  2:
        #print(flows[0].shape,im0.shape,t_value)
        loss = 0.0
        flow_0t_l = flows[0]
        flow_1t_l = flows[1]
        warped_fir = self.softsplat(im0,(1/t_value).view(-1,1,1,1)* flow_0t_l)
        warped_sec = self.softsplat(im1,(1/(1- t_value)).view(-1,1,1,1) * flow_1t_l)

        loss_f = nn.L1Loss()
        loss = loss_f(warped_fir,im1) + loss_f(warped_sec,im0)
        return loss

class set_smoothness_loss_forward(nn.Module):
    def __init__(self, args,weight=150.0, edge_aware=True):
        super(set_smoothness_loss_forward, self).__init__()
        self.edge_aware = edge_aware
        self.weight = weight ** 2
        self.args = args

    def forward(self, flow, img0,img1):
        if(self.args.fixsmoothtwistup):
            flow_01 = flow[:,:2,:] 
            flow_10 = flow[:,2:,:]
        else:
            flow_10 = flow[:,:2,:] 
            flow_01 = flow[:,2:,:]
            
        img0_gh = torch.mean(torch.pow((img0[:, :, 1:, :] - img0[:, :, :-1, :]), 2), dim=1).unsqueeze(1)#, keepdims=True)
        img0_gw = torch.mean(torch.pow((img0[:, :, :, 1:] - img0[:, :, :, :-1]), 2), dim=1).unsqueeze(1)#, keepdims=True)
        img1_gh = torch.mean(torch.pow((img1[:, :, 1:, :] - img1[:, :, :-1, :]), 2), dim=1).unsqueeze(1)#, keepdims=True)
        img1_gw = torch.mean(torch.pow((img1[:, :, :, 1:] - img1[:, :, :, :-1]), 2), dim=1).unsqueeze(1)#, keepdims=True)

        weight0_gh = torch.exp(-self.weight * img0_gh)
        weight0_gw = torch.exp(-self.weight * img0_gw)
        weight1_gh = torch.exp(-self.weight * img1_gh)
        weight1_gw = torch.exp(-self.weight * img1_gw)

        flow10_gh = torch.abs(flow_10[:, :, 1:, :] - flow_10[:, :, :-1, :])
        flow10_gw = torch.abs(flow_10[:, :, :, 1:] - flow_10[:, :, :, :-1])
        flow01_gh = torch.abs(flow_01[:, :, 1:, :] - flow_01[:, :, :-1, :])
        flow01_gw = torch.abs(flow_01[:, :, :, 1:] - flow_01[:, :, :, :-1])

        if self.edge_aware:
            return (torch.mean(weight0_gh * flow01_gh)+ torch.mean(weight1_gh * flow10_gh) 
                + torch.mean(weight0_gw * flow01_gw) + torch.mean(weight1_gw * flow10_gw)) * 0.25
        else:
            return (torch.mean(flow_gh) + torch.mean(flow_gw)) * 0.5
            
def get_batch_images(args, save_img_num, save_images):  ## For visualization during training phase
    width_num = len(save_images)
    log_img = np.zeros((save_img_num * args.patch_size, width_num * args.patch_size, 3), dtype=np.uint8)
    pred_frameT, pred_coarse_flow, pred_fine_flow, frameT, simple_mean, occ_map = save_images
    for b in range(save_img_num):
        output_img_tmp = denorm255(pred_frameT[b, :])
        output_coarse_flow_tmp = pred_coarse_flow[b, :2, :, :]
        output_fine_flow_tmp = pred_fine_flow[b, :2, :, :]
        gt_img_tmp = denorm255(frameT[b, :])
        simple_mean_img_tmp = denorm255(simple_mean[b, :])
        occ_map_tmp = occ_map[b, :]

        output_img_tmp = np.transpose(output_img_tmp.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        output_coarse_flow_tmp = flow2img(np.transpose(output_coarse_flow_tmp.detach().cpu().numpy(), [1, 2, 0]))
        output_fine_flow_tmp = flow2img(np.transpose(output_fine_flow_tmp.detach().cpu().numpy(), [1, 2, 0]),output=False)
        gt_img_tmp = np.transpose(gt_img_tmp.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        simple_mean_img_tmp = np.transpose(simple_mean_img_tmp.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        occ_map_tmp = np.transpose(occ_map_tmp.detach().cpu().numpy() * 255.0, [1, 2, 0]).astype(np.uint8)
        occ_map_tmp = np.concatenate([occ_map_tmp, occ_map_tmp, occ_map_tmp], axis=2)

        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 0 * args.patch_size:1 * args.patch_size,
        :] = simple_mean_img_tmp
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 1 * args.patch_size:2 * args.patch_size,
        :] = output_img_tmp
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 2 * args.patch_size:3 * args.patch_size,
        :] = gt_img_tmp
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 3 * args.patch_size:4 * args.patch_size,
        :] = output_coarse_flow_tmp
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 4 * args.patch_size:5 * args.patch_size,
        :] = output_fine_flow_tmp
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 5 * args.patch_size:6 * args.patch_size,
        :] = occ_map_tmp

    return log_img

# flow: [B,C,H,W]
def flow_upscale(flow,H,W):
    upscale = H/flow.shape[2]
    if(upscale != W/flow.shape[3]):
        raise exception("In flow_upscale method the scaling factors from height and width differ. Height upscale: "+ str(upscale)+ " Width upscale:" + str(W/flow.shape[3]))
    new_flow = upscale * F.interpolate(flow.detach(), size=(H,W), mode='bicubic', align_corners=False) 
    return new_flow
# H and W:  384 384                                                                                                     
# torch.Size([8, 3, 48, 48])                                                                                            
# torch.Size([8, 4, 96, 96])                                                                                            
# torch.Size([8, 4, 12, 12])                                                                                            
# torch.Size([8, 3, 384, 384])                                                                                          
# torch.Size([8, 3, 192, 192])                                                                                          
# torch.Size([8, 3, 96, 96])      
def get_pyramid_images(args,save_img_num,save_images,frameT,mean_im):
    width_num = len(save_images)+3 # plus one for ground truth,frameT and diff
    log_img = np.zeros((save_img_num * args.patch_size, width_num * args.patch_size, 3), dtype=np.uint8)
    num_im_each = len(save_images)//2
    predictions = save_images[:num_im_each]
    flows = save_images[num_im_each:]
    
    H,W = predictions[0].shape[2:4]
    #print("H and W: ",H,W)
    for i in range(len(flows)):
        flows[i] = flow_upscale(flows[i],H,W)
    for i in range(len(predictions)):
        predictions[i] = F.interpolate(predictions[i],size=(H,W),mode="bicubic",align_corners=False)

    
    for b in range(save_img_num):
        temp_flows = []
        temp_preds = []
        gt_img_tmp = denorm255(frameT[b, :])
        mean_image = denorm255(mean_im[b,:])
        for i in range(len(flows)):
            temp_flows.append(flows[i][b, :2, :, :])
            temp_flows[-1] = flow2img(np.transpose(temp_flows[-1].detach().cpu().numpy(), [1, 2, 0]))
            mi = np.min(temp_flows)
            ma = np.max(temp_flows)
            #temp_flows[-1] = ((temp_flows[-1] - mi)/(ma -mi))*255
            temp_preds.append(denorm255(predictions[i][b, :]))
            temp_preds[-1] = np.transpose(temp_preds[-1].detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        
        gt_img_tmp = np.transpose(gt_img_tmp.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)
        mean_image = np.transpose(mean_image.detach().cpu().numpy(), [1, 2, 0]).astype(np.uint8)

        preds_gray = temp_preds[0][:,:,0]*0.2989 +temp_preds[0][:,:,1]*0.5870 + temp_preds[0][:,:,2]* 0.1140 
        gt_gray = gt_img_tmp[:,:,0]*0.2989 + gt_img_tmp[:,:,1]*0.5870 + gt_img_tmp[:,:,2]* 0.1140 
        diff_pic = np.expand_dims(np.abs(preds_gray - gt_gray),axis=2)
        diff_pic = np.concatenate([diff_pic, diff_pic, diff_pic], axis=2) * 1.5

        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 0 * args.patch_size:1 * args.patch_size,
        :] = gt_img_tmp
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, 1 * args.patch_size:2 * args.patch_size,
        :] = mean_image
        for index,i in enumerate(predictions):
            log_img[(b) * args.patch_size:(b + 1) * args.patch_size, (2+index) * args.patch_size:(3+index) * args.patch_size,
        :] = temp_preds[index]
        for index,i in enumerate(flows):
            log_img[(b) * args.patch_size:(b + 1) * args.patch_size, (2+index+num_im_each) * args.patch_size:(3+index+num_im_each) * args.patch_size,
        :] = temp_flows[index]
        log_img[(b) * args.patch_size:(b + 1) * args.patch_size, (2+2*num_im_each) * args.patch_size:(3+2*num_im_each) * args.patch_size,
        :] = diff_pic

    return log_img


def get_test_pred_flow(args,fine_flow,output_img,target_img):
    width_num = 3 # plus one for ground truth,frameT and diff
    log_img = np.zeros((2 * 2160, 2 * 4096, 3), dtype=np.uint8)

    #print("Shapes: ",fine_flow.shape,output_img.shape,target_img.shape)
    
    flow_pic = fine_flow[0,:2,:]
    flow_pic= flow2img(np.transpose(flow_pic.detach().cpu().numpy(), [1, 2, 0]))


    preds_gray = output_img[:,:,0]*0.2989 +output_img[:,:,1]*0.5870 + output_img[:,:,2]* 0.1140 
    gt_gray = target_img[:,:,0]*0.2989 + target_img[:,:,1]*0.5870 + target_img[:,:,2]* 0.1140 
    diff_pic = np.expand_dims(np.abs(preds_gray - gt_gray),axis=2)
    diff_pic = np.concatenate([diff_pic, diff_pic, diff_pic], axis=2)

    log_img[0* 2160:( 1) * 2160, 0 * 4096:1 * 4096,
    :] = target_img
    log_img[0 * 2160:(1) * 2160, 1 * 4096:2 * 4096,
    :] = output_img
    log_img[1 *2160:( 2) * 2160, 0 * 4096:1 * 4096,
    :] = flow_pic
    

    return flow_pic,diff_pic


def flow2img(flow, logscale=True, scaledown=6, output=False):
    """
    topleft is zero, u is horiz, v is vertical
    red is 3 o'clock, yellow is 6, light blue is 9, blue/purple is 12
    """
    u = flow[:, :, 1]
    # u = flow[:, :, 0]
    v = flow[:, :, 0]
    # v = flow[:, :, 1]

    colorwheel = makecolorwheel()
    ncols = colorwheel.shape[0]

    radius = np.sqrt(u ** 2 + v ** 2)
    if output:
        print("Maximum flow magnitude: %04f" % np.max(radius))
        print("Mean flow magnitude: ",np.mean(radius))
    if logscale:
        radius = np.log(radius + 1)
        if output:
            print("Maximum flow magnitude (after log): %0.4f" % np.max(radius))
    radius = radius / scaledown
    if output:
        print("Maximum flow magnitude (after scaledown): %0.4f" % np.max(radius))
    # rot = np.arctan2(-v, -u) / np.pi
    rot = np.arctan2(v, u) / np.pi

    fk = (rot + 1) / 2 * (ncols - 1)  # -1~1 maped to 0~ncols
    k0 = fk.astype(np.uint8)  # 0, 1, 2, ..., ncols

    k1 = k0 + 1
    k1[k1 == ncols] = 0

    f = fk - k0

    ncolors = colorwheel.shape[1]
    img = np.zeros(u.shape + (ncolors,))
    for i in range(ncolors):
        tmp = colorwheel[:, i]
        col0 = tmp[k0]
        col1 = tmp[k1]
        col = (1 - f) * col0 + f * col1

        idx = radius <= 1
        # increase saturation with radius
        col[idx] = 1 - radius[idx] * (1 - col[idx])
        # out of range
        col[~idx] *= 0.75
        # img[:,:,i] = np.floor(255*col).astype(np.uint8)

        img[:, :, i] = np.clip(255 * col, 0.0, 255.0).astype(np.uint8)

    # return img.astype(np.uint8)
    return img


def makecolorwheel():
    # Create a colorwheel for visualization
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[col:col + RY, 0] = 1
    colorwheel[col:col + RY, 1] = np.arange(0, 1, 1. / RY)
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = np.arange(1, 0, -1. / YG)
    colorwheel[col:col + YG, 1] = 1
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 1
    colorwheel[col:col + GC, 2] = np.arange(0, 1, 1. / GC)
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = np.arange(1, 0, -1. / CB)
    colorwheel[col:col + CB, 2] = 1
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 1
    colorwheel[col:col + BM, 0] = np.arange(0, 1, 1. / BM)
    col += BM

    # MR
    colorwheel[col:col + MR, 2] = np.arange(1, 0, -1. / MR)
    colorwheel[col:col + MR, 0] = 1

    return colorwheel





def upscale_pix_transform(img,guide):

    #os.environ["OMP_PROC_BIND"] = os.environ.get("OMP_PROC_BIND", "true")


    from pixtransform.pix_transform.pix_transform import PixTransform
    from pixtransform.utils.utils import downsample,align_images
    #from prox_tv import tvgen


    ####  define parameters  ########################################################
    params = {'img_idxs' : [], # idx images to process, if empty then all of them
                
            'scaling': 4,
            'greyscale': False, # Turn image into grey-scale
            'channels': -1,
            
            'spatial_features_input': True,
            'weights_regularizer': [0.0001, 0.001, 0.0001], # spatial color head
            'loss': 'l1',
    
            'optim': 'adam',
            'lr': 0.001,
                    
            'batch_size': 32,
            'iteration': 1024*32*32//32,
                    
            'logstep': 64,
            
            'final_TGV' : False, # Total Generalized Variation in post-processing
            'align': False, # Move image around for evaluation in case guide image and target image are not perfectly aligned
            'delta_PBP': 1, # Delta for percentage of bad pixels 
            }

   
    # [C,H,W] and numpy
    predicted_target_img = PixTransform(guide_img=guide,source_img=img,params=params)

    # if params['final_TGV'] :
    #     print("applying TGV...")
    #     predicted_target_img = tvgen(predicted_target_img,[0.1, 0.1],[1, 2],[1, 1])
        
    if params['align'] :
        print("aligning...")
        target_img,predicted_target_img = align_images(target_img,predicted_target_img)

    
    if target_img is not None:
        # compute metrics and plot results
        MSE = np.mean((predicted_target_img - target_img) ** 2)
        MAE = np.mean(np.abs(predicted_target_img - target_img))
        PBP = np.mean(np.abs(predicted_target_img - target_img) > params["delta_PBP"])

        print("MSE: {:.3f}  ---  MAE: {:.3f}  ---  PBP: {:.3f}".format(MSE,MAE,PBP))
        print("\n\n")
    # img = np.array(Image.open("pixtransform/0.png"))
    # img =cv2.resize(img,dsize=(2048,2048),interpolation=cv2.INTER_CUBIC )
    # guide = img
    # print(img.shape)
    # downit = cv2.resize(img,dsize=(512,512),interpolation=cv2.INTER_AREA)
    # downit =  0.299* downit[:,:,0] + 0.587  * downit[:,:,1] + 0.114 * downit[:,:,2]
    # guide =    np.transpose(guide,(2,0,1))

    # print("shapes downit/guide ",downit.shape,guide.shape)
    # upscale_pix_transform(downit,guide)

def upscale_JBU(img,guide):
    import argparse
    #from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import Pool

    # parser = argparse.ArgumentParser(description="Perform Joint Bilateral Upsampling with a source and reference image")
    # parser.add_argument("source",default="sds", help="Path to the source image")
    # parser.add_argument("reference", default="sda",help="Path to the reference image")
    # parser.add_argument("output", default="sdad",help="Path to the output image")
    # parser.add_argument('--radius', dest='radius', default=2, help='Radius of the filter kernels (default: 2)')
    # parser.add_argument('--sigma-spatial', dest='sigma_spatial', default=2.5, help='Sigma of the spatial weights (default: 2.5)')
    # parser.add_argument('--sigma-range', dest='sigma_range', help='Sigma of the range weights (default: standard deviation of the reference image)')
    # args = parser.parse_args()
    argsradius = 2
    argssigma_spatial = 2.5

    ref_H = guide.shape[0]
    ref_W = guide.shape[1]
    # source_image = Image.open(args.source)

    # reference_image = Image.open(args.reference)
    # reference = np.array(reference_image)

    # source_image_upsampled = source_image.resize(reference_image.size, Image.BILINEAR)
    # source_upsampled = np.array(source_image_upsampled)
    source_upsampled = cv2.resize(img,dsize=(ref_W,ref_H),interpolation=cv2.INTER_NEAREST)
    reference = guide

    scale = source_upsampled.shape[1] / reference.shape[1]
    radius = int(argsradius)
    diameter = 2 * radius + 1
    step = int(np.ceil(1 / scale))
    padding = radius * step
    sigma_spatial = float(argssigma_spatial)
    sigma_range =  np.std(reference) #float(args.sigma_range) if args.sigma_range else np.std(reference)

    reference = np.pad(reference, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)
    source_upsampled = np.pad(source_upsampled, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)

    # Spatial Gaussian function.
    x, y = np.meshgrid(np.arange(diameter) - radius, np.arange(diameter) - radius)
    kernel_spatial = np.exp(-1.0 * (x**2 + y**2) /  (2 * sigma_spatial**2))
    kernel_spatial = np.repeat(kernel_spatial, 3).reshape(-1, 3)

    # Lookup table for range kernel.
    lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * sigma_range**2))

    

    #executor = ProcessPoolExecutor()
    #result = executor.map(process_row, range(ref_H))
    #executor.shutdown(True)
    start_time = time.time()
    with Pool(24) as p:
        result = p.starmap(process_row,[(i,padding,reference,step,source_upsampled,lut_range,kernel_spatial,ref_W) for i in range(ref_H)])
    print("time for execution: ",time.time()-start_time)
    #print(result)
    #print("output shape: ",np.array(list(result)).astype(np.uint8).shape)
    start_time = time.time()
    result = np.array(list(result)).astype(np.uint8)
    Image.fromarray(result).save("upsamplingresults/out.png")
    print("time for conversion: ",time.time()-start_time)

    return result
   

def process_row(y,padding,reference,step,source_upsampled,lut_range,kernel_spatial,ref_W):
        result = np.zeros((ref_W, 3))
        y += padding
        for x in range(padding, reference.shape[1] - padding):
            I_p = reference[y, x]
            patch_reference = reference[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)
            patch_source_upsampled = source_upsampled[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)

            kernel_range = lut_range[np.abs(patch_reference - I_p).astype(int)]
            weight = kernel_range * kernel_spatial
            k_p = weight.sum(axis=0)
            result[x - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=0) / k_p)

        return result

def get_gaussian_kernel_weights(kernel_size=3, sigma=2, channels=3):
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

	    # # Reshape to 2d depthwise convolutional weight
	    # gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
	    # gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

	    # gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
	    #                             kernel_size=kernel_size, groups=channels, bias=False,padding=1)# zero padding!

	    # gaussian_filter.weight.data = gaussian_kernel
	    # gaussian_filter.weight.requires_grad = False
	    
	    return gaussian_kernel

def my_jbu(img,guide):
    parts = []
    for color in range(3):
        
        H,W,_ = guide.shape
        C = 1
        kernelsize = 3
        
        im_local = (torch.as_tensor(img[:,:,color:color+1])/255.).permute(2,0,1).unsqueeze(0).to(0)
        guide_local = (torch.as_tensor(guide[:,:,color:color+1])/255.).permute(2,0,1).unsqueeze(0).to(0)
        #print(guide_local.shape,im_local.shape)
        imtem =  F.interpolate(im_local,scale_factor=guide_local.shape[2]//im_local.shape[2],mode="nearest")
        
        #
        # (N,CxKernelsize,L)
        toblockfun = nn.Unfold(kernel_size=kernelsize,stride=1,padding=1)
        gaussweights = get_gaussian_kernel_weights(kernel_size=kernelsize,sigma=0.5).to(0)
        
        start_time = time.time()
        print("Imtem: ",imtem.shape)
        folded_im = toblockfun(imtem).squeeze(0).reshape(C,kernelsize**2,-1)

        
        rangeimg = torch.cat([torch.abs(folded_im[:,i:i+1,:] - folded_im[:,4:5,:]) for i in range(kernelsize**2)],dim=1)
        # here comes gauss
        rangeimg = rangeimg/torch.sum(rangeimg,dim=1,keepdim=True)
        rangeimg = rangeimg * folded_im
        
        folded_im = 0
        imtem = 0
        im_local = 0
        torch.cuda.empty_cache()
        
        folded_guide = toblockfun(guide_local).squeeze(0).reshape(C,kernelsize**2,-1)
        print(folded_guide.shape,gaussweights.shape)
        #time.sleep(3)
        rangeimg = folded_guide * gaussweights.view(1,-1,1) * rangeimg
        #time.sleep(3)

        folded_guide = 0
        torch.cuda.empty_cache()
        # print("foldedimg: ",folded_im.shape)
        # print("rangeimg: ",rangeimg.shape)
        # print("gaussimg: ",gaussimg.shape)
        outimage = torch.sum( rangeimg ,dim=1).reshape(C,H,W)
        parts.append(outimage)
        print("til here: ",time.time()-start_time)
        #blocked = imtem.reshape(-1,blocks_y,blocks_x).permute(0,2,1).reshape(chan,wiS**2,blocks_x,blocks_y).permute(0,2,3,1).reshape(chan,blocks_x,blocks_y,wiS,wiS)
    outimage = torch.cat(parts,dim=1)
        #sys.exit()

if __name__ == "__main__":
    img = np.array(Image.open("pixtransform/0.png"))
    print("shape pre trans: ",img.shape)
    dofac = 1
    img =cv2.resize(img,dsize=(4096//dofac,2160//dofac),interpolation=cv2.INTER_AREA )
    guide = img

    downit = cv2.resize(img,dsize=(1024//dofac,540//dofac),interpolation=cv2.INTER_AREA)
    upit = cv2.resize(downit,dsize=(4096//dofac,2160//dofac),interpolation=cv2.INTER_CUBIC)
    #downit =  0.299* downit[:,:,0] + 0.587  * downit[:,:,1] + 0.114 * downit[:,:,2]
    #guide =    np.transpose(guide,(2,0,1))


    print("shapes downit/guide ",downit.shape,guide.shape)

    start_time_gl = time.time()
    retpic =my_jbu(downit,guide) 
    print("Time needed: ",time.time()-start_time_gl)
    stats = torch.cuda.memory_stats()
    peak_bytes_requirement = stats["allocated_bytes.all.peak"]
    print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
    sys.exit()

    retpic = np.transpose(retpic,(2,1,0))
    guide = np.transpose(guide,(2,1,0))
    upit = np.transpose(upit,(2,1,0))

    print(retpic.shape,guide.shape)
    psnr = peak_signal_noise_ratio(retpic,guide)
    psnrupit = peak_signal_noise_ratio(upit,guide)
    print("PSNR: ",psnr," / ",psnrupit)