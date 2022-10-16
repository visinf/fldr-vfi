import os, pathlib, shutil, torch,cv2,glob
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F
import torch.nn.functional as FF
import matplotlib.pyplot as plt
from myAdditions import MyPWC
import torch.utils.data as data
import torch.nn as nn
# from filtered_dataset.model.pwc import PWC, compute_flow
# from filtered_dataset.utils.warping import backwarp
def RGBframes_np2Tensor(imgIn, channel):
    ## input : T, H, W, C
   
    # to Tensor
    ts = (channel, 0, 1, 2)  ############# dimension order should be [C, T, H, W]
    imgIn = torch.Tensor(imgIn.transpose(ts).astype(float)).mul_(1.0)

    # normalization [0,1]
    imgIn = ((imgIn / 255.0) *2)-1 

    return imgIn

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
   
    """ np2Tensor [0,1] normalized """
    frames = RGBframes_np2Tensor(frames, 3)

    return frames

class Xiph_Test(data.Dataset):
    def __init__(self, args, validation,twoKC=False):
        self.args = args
        self.framesPath = []
        self.twoKC = twoKC
        assert not validation 
        counter = -1
        path = "XiphDataset/netflix"
        for strFile in ['BoxingPractice', 'Crosswalk', 'DrivingPOV', 'FoodMarket', 'FoodMarket2', 'RitualDance', 'SquareAndTimelapse', 'Tango']:
            for intFrame in range(2, 99, 2):
                if(validation):
                    counter += 1
                    if(counter % 19 != 0):
                        continue
                npyFirst = os.path.join(path, strFile + '-' + str(intFrame - 1).zfill(3) + '.png')
                npySecond = os.path.join(path,  strFile + '-' + str(intFrame + 1).zfill(3) + '.png')
                npyReference = os.path.join(path,  strFile + '-' + str(intFrame).zfill(3) + '.png')
                self.framesPath.append([npyFirst,npySecond,npyReference])


        self.num_scene = len(self.framesPath)  # total test scenes
        if len(self.framesPath) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + path + "\n"))
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
        #if(self.twoKC):
        #frames = frames[:,:,540:-540,1024:-1024]
        #assert list(frames.shape) == [3,3,1080,2048]
        # if(self.args.xiph2k):
        #     frames = F.interpolate(frames,scale_factor=1/2,mode="bilinear",align_corners=self.args.align_cornerse)
        return frames#, np.expand_dims(np.array(0.5, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.num_scene

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

class Inter4K_Test(data.Dataset):
    def __init__(self, args, multiple=8,twoK=False,scenerange=8):
        self.args = args
        self.twoK = False
        self.multiple = 8
        self.scenerange = scenerange
        assert self.scenerange%self.multiple == 0
        self.testPath = []
        # Return """ make [I0,I1,It,t,scene_folder] """
        testfolder = "inter4K/Inter4KNewTestset/"
        #folders = os.listdir(testfolder)
        folders = sorted(os.listdir(testfolder),key=lambda x: int(x))
        print(folders)
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
        # if(self.twoK):
        #     frames = F.interpolate(frames,scale_factor=1/2,mode="bilinear",align_corners=self.args.align_cornerse)
        return frames, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


class X_Test(data.Dataset):
    def __init__(self, args, multiple=8, validation=False,twoKC=False):
        self.args = args
        self.multiple = multiple
        self.validation = validation
        self.twoKC = twoKC
        path = superprefix = './../../../' + 'X-Train/' + "test"
        # Return """ make [I0,I1,It,t,scene_folder] """
        self.testPath = make_2D_dataset_X_Test(path, multiple, t_step_size=32)

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
        frames = frames_loader_test(self.args, I0I1It_Path, False)
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
        

        # if(self.args.xtest2k):
        #     frames = F.interpolate(frames,scale_factor=1/2,mode="bilinear",align_corners=self.args.align_cornerse)
        # if(self.twoKC):
        #frames = frames[:,:,540:-540,1024:-1024]
        #assert list(frames.shape) == [3,3,1080,2048]
        return frames#, np.expand_dims(np.array(t_value, dtype=np.float32), 0), scene_name, [It_path, I0_path, I1_path]

    def __len__(self):
        return self.nIterations


def bwarp(device, x, flo,withmask=True):
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
            grid = grid.to(device)
        vgrid = torch.autograd.Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)  # [B,H,W,2]
        output = nn.functional.grid_sample(x, vgrid)#, align_corners=True)
        mask = torch.autograd.Variable(torch.ones(x.size())).to(device)
        mask = nn.functional.grid_sample(mask, vgrid)#, align_corners=True)

        # mask[mask<0.9999] = 0
        # mask[mask>0] = 1
        mask = mask.masked_fill_(mask < 0.999, 0)
        mask = mask.masked_fill_(mask > 0, 1)

        if(withmask):
            return output * mask
        else:
            return output


#get_flow(self,im0,im1):
def main(opts):
    # Remove raw in directory for new directory in case no output root is given




    # Get optical flow model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    flow_model = MyPWC(opts) 
    flow_model.flow_predictor.eval()
    out_root = "flowcomps"

    percentiles = [1,25,50,75,99]
    data_test = X_Test(opts)#X_Test(opts)#Xiph_Test(opts,validation=False)   #Inter4K_Test(opts,scenerange=8)
    dataloader = torch.utils.data.DataLoader(data_test, batch_size=1, drop_last=False, shuffle=False, pin_memory=False)
    # Get directories with sequences
    glob_frame_idx = 0
    flow_image_mean = []
    # flow_pixelwise = []
    l1_diff_image_mean = []
    flow_diff_image_mean = []
    flows_mag_threshold_higher = [[]for i in percentiles]

    whole_flow = 0
    # Compute optical flows for the sequence
    for overindex,f_idx in enumerate(dataloader):

        if(overindex%7 != 3):
            continue
        print(overindex//7)
        # if(overindex%10!=0):
        #     continue
        
        im0 = f_idx[:,:,0,:].to(device)
        im1 = f_idx[:,:,1,:].to(device)
        #imt = f_idx[:,:,2,:].to(device)
        #print(im0.shape,im1.shape)
        im0 = FF.interpolate(im0,scale_factor=1/2,mode="bilinear",align_corners=False)
        im1 = FF.interpolate(im1,scale_factor=1/2,mode="bilinear",align_corners=False)
        #print(im0.shape,im1.shape)
        flow = 2 * flow_model.get_flow(im0,im1).detach().cpu()[:,:2].squeeze(0) #[flow10,flow01]
        #print(flow.shape)
        flow = FF.interpolate(flow.unsqueeze(0),scale_factor=2,mode="bilinear",align_corners=False).squeeze(0)
        #print(flow.shape)
        #print("flowshape: ",flow.shape)

        # Compute motion magnitude
        flow_mag = torch.sqrt(flow[0]**2 + flow[1]**2)
        # Compute l1 difference between reference and warped frame
        # f0_warped = bwarp("cpu",im1.cpu(), flow.unsqueeze(0))
        # l1_diff = torch.abs(im0.cpu() - f0_warped).mean()
        # Compute difference between motion fields

        
        #fmin1 = Image.open(fmin1_path)
        #fmin1 = imt.unsqueeze(0)
        #flow_min1 = flow_model.get_flow(im0,im1).detach().cpu()[:,:2].detach().cpu().squeeze(0)
        #flow_min1 = compute_flow(flow_model, f0.to(device), fmin1.to(device)).squeeze(0).detach().cpu()
        #flow_diff = torch.abs(flow - flow_min1)
        #flow_diff_mean = torch.sqrt(flow_diff[0] ** 2 + flow_diff[1] ** 2).mean()

        #print(flow_mag.flatten().sort())
        # Compute a flow threshold where (1-percentile) of the pixels are higher than this threshold
        if(glob_frame_idx==0):
            whole_flow = flow_mag.flatten()#.sort()[0]
            #print("ONCE: ",whole_flow.shape)
        else:
            whole_flow =  torch.cat([whole_flow,flow_mag.flatten()])#.sort()[0]
            #print("other: ",whole_flow.shape)

        # flow_mag_sorted = flow_mag.flatten().sort()[0]
        # for index,i in enumerate(percentiles):
        #     #print(len(flow_mag.flatten().sort()[0]),int(flow_mag_sorted.numel() * i/100))
        #     flows_mag_threshold_higher[index].append(flow_mag_sorted[int(flow_mag_sorted.numel() * i/100)].item())
        
        # print(flows_mag_threshold_higher[0][-1]," | ",flows_mag_threshold_higher[1][-1]," | ",flows_mag_threshold_higher[2][-1]," | ",flows_mag_threshold_higher[3][-1]," | ",flows_mag_threshold_higher[4][-1])

        # Store values for histograms
        # flow_image_mean.append(flow_mag.mean().item())
        # # flow_pixelwise.append(flow_mag.numpy())
        # l1_diff_image_mean.append(l1_diff.item())
        #flows_mag_threshold_higher.append(flow_mag_threshold_higher.item())
       
        #flow_diff_image_mean.append(flow_diff_mean.item())
        print(glob_frame_idx)
        glob_frame_idx += 1
        # if(glob_frame_idx>1):
        #     break

    whole_flow_sorted = whole_flow.sort()[0]
    for index,i in enumerate(percentiles):
        print("Percentile: ",i/100,whole_flow_sorted[int(whole_flow_sorted.numel() * i/100)].item())

    # for i in flows_mag_threshold_higher:
    #     print(np.mean(np.array(i)))

    # 1. Histogram for image mean flow frequency
    plt.figure(0)
    plt.hist(flow_image_mean, bins=opts.num_bins)
    plt.xlabel('Flow Magnitude')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_root, "image_mean_flow_frequency.png"))

    # # 2. Histogram for pixelwise flow frequency
    # plt.figure(1)
    # plt.hist(np.array(flow_pixelwise).flatten(), bins=opts.num_bins)
    # plt.xlabel('Flow Magnitude')
    # plt.ylabel('Frequency')
    # plt.savefig(os.path.join(out_root, "pixel_wise_flow_frequency.png"))

    # 3. Histogram for l1 difference of image intensities
    plt.figure(2)
    plt.hist(l1_diff_image_mean, bins=opts.num_bins)
    plt.xlabel('L1 Difference Image Intensity')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_root, "l1_diff_image_intensity.png"))

    # # 4. Histogram for difference of between motion fields
    # plt.figure(3)
    # plt.hist(flow_diff_image_mean, bins=opts.num_bins)
    # plt.xlabel('Magnitude of Motion Differences')
    # plt.ylabel('Frequency')
    # plt.savefig(os.path.join(out_root, "image_mean_magnitude_motion_diff.png"))

    # 5. Histogram for flow threshold where 5 % of pixels are higher
    plt.figure(4)
    plt.hist(flows_mag_threshold_higher, bins=opts.num_bins)
    plt.xlabel('Magnitude of Motion Threshold (95 % Percentile)')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(out_root, "motion_magnitude_threshold_95_percentile.png"))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--in-root",
        type=str,
        default="/Users/sherwinbahmani/Desktop/dataset/full_scaled",
        help="Use either the scaled or cropped dataset"
             "/full_{scaled,cropped}/{train, val, test}/{00000, 00001, ...}/{both, left, right}/"
             "[00000.jpg, 00001.jpg, 00002.jpg]"
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="/full_{scaled,cropped}_visualize output with flows"
    )
    parser.add_argument(
        "--keep-dir",
        action="store_true",
        help="Don't delete output directory before creating sequences"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=0.95,
    )
    clargs = parser.parse_args()
    with torch.no_grad():
        main(clargs)