# This code contains parts of XVFInet from Sim et al. (https://github.com/JihyongOh/XVFI) 
# Their extensive code and Dataset were crucial for this.

from __future__ import division
#from asyncio import FastChildWatcher
import os, glob, sys, torch, shutil, random, math, time, cv2
from tracemalloc import start
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from torch.nn import init

from skimage.metrics import structural_similarity,peak_signal_noise_ratio
from torch.autograd import Variable
from torchvision import models

from useful import torch_prints


from skimage.transform import rescale


from softSplat import Softsplat
from PIL import Image
class save_manager():
    def __init__(self, args):
        self.args = args


        self.model_dir = self.args.net_type + '_' + self.args.dataset + '_exp' + str(self.args.exp_num)
        print("model_dir:", self.model_dir)


        self.checkpoint_dir = os.path.join(self.args.checkpoint_dir, self.model_dir)

        check_folder(self.checkpoint_dir)

        print("checkpoint_dir:", self.checkpoint_dir)

        self.text_dir = os.path.join(self.args.text_dir, self.model_dir)
        print("text_dir:", self.text_dir)

        """ Save a text file """
        if (not os.path.exists(self.text_dir + '.txt') or not args.continue_training) and not args.phase=='test':
            self.log_file = open(self.text_dir+ '.txt', 'w')

            self.log_file.write('----- Model parameters -----\n')
            self.log_file.write(str(datetime.now())[:-7] + '\n')
            for arg in vars(self.args):
                self.log_file.write('{} : {}\n'.format(arg, getattr(self.args, arg)))

            self.log_file.close()



    def write_info(self, strings):
        self.log_file = open(self.text_dir+ '.txt', 'a')
        self.log_file.write(strings)
        self.log_file.close()

    def save_best_model(self, combined_state_dict, best_PSNR_flag):
        file_name = os.path.join(self.checkpoint_dir, self.model_dir + '_latest.pt')

        torch.save(combined_state_dict, file_name)
        if best_PSNR_flag:
            shutil.copyfile(file_name, os.path.join(self.checkpoint_dir, self.model_dir + '_best_PSNR.pt'))



    def save_epc_model(self, combined_state_dict, epoch):
        file_name = os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch) + '.pt')

        torch.save(combined_state_dict, file_name)

    def load_epc_model(self, epoch):
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch - 1) + '.pt'))
        print("load model '{}', epoch: {}, best_PSNR: {:3f}".format(
            os.path.join(self.checkpoint_dir, self.model_dir + '_epc' + str(epoch - 1) + '.pt'), checkpoint['last_epoch'] + 1,
            checkpoint['best_PSNR']))
        return checkpoint

    def load_model(self,specific=-1 ,takeBestModel=False):

        suffix =  '_latest.pt'
        if(specific != -1):
            suffix = '_epc' + str(specific) + '.pt'
        elif(takeBestModel):
            suffix = '_best_PSNR.pt'
        print(os.path.join(self.checkpoint_dir, self.model_dir +suffix))
        checkpoint = torch.load(os.path.join(self.checkpoint_dir, self.model_dir + suffix), map_location="cuda:"+str(self.args.gpu))

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

        if hasattr(m, 'bias') and m.bias is not None:
            init.zeros_(m.bias)


def get_train_data(args, max_t_step_size,device):
    if args.dataset == 'X4K1000FPS':
        data_train = X_Train(args, max_t_step_size,device)
    elif args.dataset == 'Vimeo':
        data_train = Vimeo_Train(args)
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=args.batch_size, drop_last=True, shuffle=True,
                                             num_workers=int(args.num_thrds), pin_memory=args.pin_memory_train,prefetch_factor=3)
    return dataloader,(None,None)


def get_test_data(args,dataset, multiple, validation,specific=""):

    if(len(specific)>0):
        dataset = specific
    if dataset == 'X4K1000FPS' and args.phase != 'test_custom':
        data_test = X_Test(args, multiple, validation)
    elif dataset == 'Vimeo' and args.phase != 'test_custom':
        data_test = Vimeo_Test(args, validation)
    elif dataset == "Xiph" :
        data_test = Xiph_Test(args,validation)
    elif dataset == "Xiph2KC" :
        data_test = Xiph_Test(args,validation,twoKC=True)
    elif dataset == "Inter4K88":
        data_test = Inter4K_Test(args,multiple,scenerange=8)  
    elif dataset == "Inter4K816":
        data_test = Inter4K_Test(args,multiple,scenerange=16) 
    

    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, drop_last=False, shuffle=False, pin_memory=args.pin_memory_test)
    return dataloader





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


        self.num_scene = len(self.framesPath)  
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
        


        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

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

        if(Xiph):
            frame = cv2.imread(path,flags=-1)
        else:
            frame = cv2.imread(path)
        frames.append(frame)
    (ih, iw, c) = frame.shape
    frames = np.stack(frames, axis=0)  # (T, H, W, 3)


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

    if channel == 1:

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



def frames_loader_train_inter4k(args, candidate_frames, frameRange):

    frames = []
    if(args.patch_size>512):
        res_choices = ["im2k.png","im4k.png"]
    else:
        res_choices = ["im1k.png","im2k.png","im4k.png"]
    ranval = random.randint(0,len(res_choices)-1)
    res = res_choices[ranval]
    for frameIndex in frameRange:
        frame =np.array(Image.open(os.path.join(candidate_frames[frameIndex],res)))

        frames.append(frame)





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

    frames = []
    for frameIndex in frameRange:
        frame =np.array(Image.open(candidate_frames[frameIndex]))
        frames.append(frame)


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
    def __init__(self, args, max_t_step_size,device):
        self.args = args
        self.max_t_step_size = max_t_step_size
        self.device = device
        self.framesPath = make_2D_dataset_X_Train(self.args.x_train_data_path)
        self.nScenes = len(self.framesPath)

        self.psnr_bil =  AverageClass('PSNR bil:', ':.4e')
        self.psnr_dct = AverageClass('PSNR dct:', ':.4e')
        # Raise error if no images found in x_train_data_path.
        if self.nScenes == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.args.x_train_data_path + "\n"))

    def __getitem__(self, idx):
        t_step_size = random.randint(2, self.max_t_step_size)
        t_list = np.linspace((1 / t_step_size), (1 - (1 / t_step_size)), (t_step_size - 1))

        candidate_frames = self.framesPath[idx]
        firstFrameIdx = random.randint(0, (64 - t_step_size))
        interIdx = random.randint(1, t_step_size - 1)  
        interFrameIdx = firstFrameIdx + interIdx  # absolute index
        t_value = t_list[interIdx - 1]  # [0,1]

        ########    TEMPORAL DATA AUGMENTATION      ####################
        if (random.randint(0, 1)):
            frameRange = [firstFrameIdx, firstFrameIdx + t_step_size, interFrameIdx]
        else:  ## temporally reversed order
            frameRange = [firstFrameIdx + t_step_size, firstFrameIdx, interFrameIdx]
            interIdx = t_step_size - interIdx 
            t_value = 1.0 - t_value

        ##########      SPATIAL DATA AUGMENTATION       ################################
        frames = frames_loader_train(self.args, candidate_frames,
                                     frameRange)  # including "np2Tensor [-1,1] normalized"


        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0)

    def __len__(self):
        return self.nScenes




def make_2D_dataset_X_Test(dir, multiple, t_step_size):

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




class Inter4K_Test(data.Dataset):
    def __init__(self, args, multiple,twoK=False,scenerange=8):
        self.args = args
        self.twoK = False
        self.multiple = 8
        self.scenerange = scenerange
        assert self.scenerange%self.multiple == 0
        self.testPath = []
        testfolder = self.args.inter4k_data_path
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

            for scenindex,scen in enumerate(scenes):
                if(not len(scen)< self.scenerange+1):
                    temscen = [os.path.join(tempath,file) for file in scen]
                    for temk in range(self.multiple-1):
                        self.testPath.append([temscen[0],temscen[self.scenerange],temscen[(temk+1)*(self.scenerange//self.multiple)],t[temk],tempath+"_scene_"+str(scenindex)])

   
            

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
            self.testPath = make_2D_dataset_X_Test(self.args.x_val_data_path, multiple, t_step_size=32)
        else:  ## test
            self.testPath = make_2D_dataset_X_Test(self.args.x_test_data_path, multiple, t_step_size=32)

        self.nIterations = len(self.testPath)

        # Raise error if no images found in test_data_path.
        if len(self.testPath) == 0:
            if validation:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.x_val_data_path + "\n"))
            else:
                raise (RuntimeError("Found 0 files in subfolders of: " + self.args.x_test_data_path + "\n"))

    def __getitem__(self, idx):
        I0, I1, It, t_value, scene_name = self.testPath[idx]

        I0I1It_Path = [I0, I1, It]

        # Returns all frames (65)
        frames = frames_loader_test(self.args, I0I1It_Path, self.validation)
        # including "np2Tensor [-1,1] normalized"

        I0_path = I0.split(os.sep)[-1]
        I1_path = I1.split(os.sep)[-1]
        It_path = It.split(os.sep)[-1]

        

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
        if(False):
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
        self.softsplat = Softsplat()
# Img: [B, C, T, H, W] 
    def forward(self, flows,im0, im1,t_value):
        # 0 -> t :2
        # 1 -> t  2:
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
            
        img0_gh = torch.mean(torch.pow((img0[:, :, 1:, :] - img0[:, :, :-1, :]), 2), dim=1).unsqueeze(1)
        img0_gw = torch.mean(torch.pow((img0[:, :, :, 1:] - img0[:, :, :, :-1]), 2), dim=1).unsqueeze(1)
        img1_gh = torch.mean(torch.pow((img1[:, :, 1:, :] - img1[:, :, :-1, :]), 2), dim=1).unsqueeze(1)
        img1_gw = torch.mean(torch.pow((img1[:, :, :, 1:] - img1[:, :, :, :-1]), 2), dim=1).unsqueeze(1)

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

def get_pyramid_images(args,save_img_num,save_images,frameT,mean_im):
    width_num = len(save_images)+3 # plus one for ground truth,frameT and diff
    log_img = np.zeros((save_img_num * args.patch_size, width_num * args.patch_size, 3), dtype=np.uint8)
    num_im_each = len(save_images)//2
    predictions = save_images[:num_im_each]
    flows = save_images[num_im_each:]
    
    H,W = predictions[0].shape[2:4]

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
    v = flow[:, :, 0]


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


        img[:, :, i] = np.clip(255 * col, 0.0, 255.0).astype(np.uint8)

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



