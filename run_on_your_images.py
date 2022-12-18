# This code contains parts of XVFInet from Sim et al. (https://github.com/JihyongOh/XVFI) 
# Their extensive code and Dataset were crucial for this.

import argparse, os, torch, cv2,  torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np

from pca_comp import DCTParams,to_pca 
from useful import getmodelconfig

from torch.autograd import Variable
from torchvision import utils
from utils import *
from fLDRnet import *
from collections import Counter
import sys
from skimage.transform import rescale





def main():
    
    
    model_net, device, args = prepare_model()
    

    ########    Short X_Test evaluation    ##########
    
    # Get image paths
    direclist = os.listdir("X_test")
    allscenes = []
    for i in direclist:
        allscenes += [os.path.join("X_test",i,k) for k in os.listdir(os.path.join("X_test",i))]


    # iterate over all images and interpolate
    psnrs = []
    for i in allscenes:
        allimgs = sorted(os.listdir(i),key=lambda x: int(x.split(".")[0]))
        for t_val in [0.125,0.25,0.375,0.5,0.625,0.75,0.875]:

            frames = load_trans_frames(i+"\\"+allimgs[0],  i+"\\"+allimgs[-1], i+"\\"+allimgs[int(t_val*32)])
            
            # execute model, but also provide ground truth image for direct evaluation if evalit is true!
            ret_psnr = run_on_images(model_net,args,device,frames,torch.tensor([[t_val]]),"GenFrames/temptest",evalit=True)

            psnrs.append(ret_psnr)
            print("PSNR: ",np.mean(psnrs))


def prepare_model():
    args = args_config()

    device =  torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu') 

    """ Initialize a model """
    model_net = args.net_object(args).apply(weights_init).to(device) 
    SM = save_manager(args)
    cudnn.benchmark = True
    
    # Fetches LATEST Model
    checkpoint = SM.load_model(takeBestModel=True,specific=args.specificCheckpoint)
    model_net.load_state_dict(checkpoint['state_dict_Model'])

    # Apply params
    params = []
    for i in range(len(args.scales)):    
        params.append(DCTParams(wiS=8,components_fraction=1/4,data_used=0.5 ) )        
    model_net.save_params(params)
    return model_net, device, args


# im0,im1,imt: paths to either image
def load_trans_frames(im0,im1,imt):
    image0 = cv2.imread(im0)
    image1 = cv2.imread(im1)
    imaget = cv2.imread(imt)
    arr = [image0,image1,imaget]
    for i in range(3):
        # to [1,-1] and C,H,W
        arr[i] = ((torch.as_tensor(arr[i])/255)*2-1).permute(2,0,1)#.unsqueeze(0)

    # to B,C,T,H,W  || T = 3 im0,im1,imt
    return torch.stack(arr,axis=0).permute(1,0,2,3).unsqueeze(0)


# model_net:  initalized model
# args:       argument parsed
# device:     torch device
# frames:     B,C,T,H,W with T = 3 || im0,im1,imt
# t_value:    t_value in [0,1]. 
# resfold:    folder where the generated image is saved
def run_on_images( model_net, args, device,frames,t_value,resfold,evalit=True):



    torch.backends.cudnn.enabled = True 

    torch.backends.cudnn.benchmark = False

    # switch to evaluate mode
    model_net.eval()


    with torch.no_grad():
        
        # Pick frameT and input_frames
        if evalit:
            frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            input_frames = frames[:, :, :-1, :, :]
        else:
            input_frames = frames[:, :, :2, :, :]
        torch.cuda.empty_cache()
        B, C, T, H, W = input_frames.size()
        _,_,_,OH,OW = input_frames.size()
        

        # reshape needed for padding
        #temp_shap = input_frames.shape
        #input_frames = input_frames.reshape(temp_shap[0],-1,temp_shap[3],temp_shap[4])
        input_frames = input_frames.reshape(B,-1,H,W)

        # Find out how much you have to pad
        div_pad = (2**args.S_tst)*8 if(args.phase=="test") else (2**args.S_trn)*8
        H_padding = (div_pad - H % div_pad) % div_pad
        W_padding = (div_pad - W % div_pad) % div_pad

        # Pad it accordingly
        input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), args.padding)
        input_frames = input_frames.reshape(B, C, T, OH+H_padding, OW+W_padding)    
        #input_frames = input_frames.reshape(temp_shap[0],temp_shap[1],temp_shap[2],temp_shap[3]+H_padding,temp_shap[4]+W_padding)    
        
        # Prepare empty list off tensors to be filled with PCA transformed images
        input_gpuList = []
        for l in range(6):
            input_gpuList.append( torch.zeros((B,int(args.img_ch*2*(8**2)*0.25) ,H//8,W//8),device=device))


        # The normal images used for warping etc
        B,C,T,H,W = input_frames.shape
        input_gpu = [F.interpolate(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W), scale_factor=args.scales[0]/ (args.scales[i]),mode='bicubic', 
            align_corners=args.align_cornerse).to(device).reshape(B,T,C,int(H*(args.scales[0]/ (args.scales[i]))),int(W*(args.scales[0]/ ( args.scales[i])))).permute(0,2,1,3,4) if(i!=0 ) else input_frames.to(device)  for i in range(args.S_tst+1)]
        

        # point in time between the input frames to be interpolated to.
        t_value = Variable(t_value.to(device))
        torch.cuda.empty_cache()

        # Actual Modelrun
        pred_frameT,_ = model_net(input_gpuList, t_value,normInput=[im.clone() for im in input_gpu],is_training=False,validation=False)
        
        
        # To numpy and cutoff
        pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())[:,:OH,:OW]
        if evalit:
            frameT = np.squeeze(frameT.detach().cpu().numpy())
        
        """ compute PSNR & SSIM """ # From output And Ground Truth!
        output_img = np.around(denorm255_np(np.transpose(pred_frameT, [1, 2, 0])))  # [h,w,c] and [-1,1] to [0,255]
        if evalit:
            target_img = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]

    
        # save generated image
        testpredspath = os.path.join(resfold) 
        check_folder(testpredspath)
        cv2.imwrite(os.path.join(testpredspath,str(int(t_value*8))+".png"),output_img)

        # PSNR evaluation       
        if evalit:
            test_psnr = psnr(target_img, output_img,args)
            #test_ssim = ssim_bgr(target_img, output_img)  
            return test_psnr
        else:
            return 0

       


def args_config():
    args = parse_args()
    args.papermodel = True
    args.exp_num = 1
    args.test5scales = True


    if(args.papermodel):
        getmodelconfig(args)

    args.fractions = [4, 16, 64, 256, 1024,4096]
    args.scales = [8,16,32,64,128,256]
    args.moreTstSc = True
    args.phase = "test"
    args.S_tst = 5
    args.fractions = [int(i) for i in args.fractions]
    args.scales = [int(i) for i in args.scales]
    args.dctvfi_nf = args.scales[0]**2//args.fractions[0]
    args.padding = "reflect" 
    args.takeBestModel = True
    return args

def parse_args():
    desc = "PyTorch implementation for XVFI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--net_type', type=str, default='fLDRnet', choices=['fLDRnet'], help='The type of Net')
    parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
    

    parser.add_argument('--text_dir', type=str, default='./text_dir', help='text_dir path')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir', help='checkpoint_dir')
    parser.add_argument('--dataset', default='X4K1000FPS', choices=['X4K1000FPS', 'Vimeo',"Inter4K",'Xiph-4K'],
                        help='Training/test Dataset')

    

    ################################        My Hyperparameters      #############################################

    parser.add_argument('--test5scales', action="store_true", help='faster?')


    ###########################         DCT-NET Hyperparameters         ###########################################
    # Information about model
    parser.add_argument('--parameters',  type=int, default=-1, help='My Big DCTNET') 
    parser.add_argument('--save_images', action="store_true", help='faster?')
        

    # FORWARD WARPING & Paper general
    parser.add_argument('--softsplat', action="store_true", help='My Big DCTNET')
    parser.add_argument('--ownsmooth', action="store_true", help='My Big DCTNET')
    parser.add_argument('--forwendflowloss', action="store_true", help='My Big DCTNET')
    parser.add_argument('--ownoccl', action="store_true", help='My Big DCTNET')
    parser.add_argument('--sminterp', action="store_true", help='My Big DCTNET')
    parser.add_argument('--sminterpWT', action="store_true", help='My Big DCTNET')
    parser.add_argument('--tparam',  type=float, default=1, help='no features inputted at refinement step')
    parser.add_argument('--noResidAddup', action="store_true", help='My Big DCTNET')
    parser.add_argument('--cutoffUnnec', action="store_true", help='My Big DCTNET')
    parser.add_argument('--fixsmoothtwistup', action="store_true", help='My Big DCTNET')
    parser.add_argument('--impmasksoftsplat', action="store_true", help='My Big DCTNET')
    parser.add_argument('--TOptimization', action="store_true", help='My Big DCTNET')
    parser.add_argument('--sminterpInpIm', action="store_true", help='My Big DCTNET')
    parser.add_argument('--tempAdamfix', action="store_true", help='My Big DCTNET')
    parser.add_argument('--simpleEVs', action="store_true", help='My Big DCTNET')
    parser.add_argument('--smallenrefine', action="store_true", help='My Big DCTNET')
    parser.add_argument('--interpOrigForw', action="store_true", help='My Big DCTNET')
    parser.add_argument('--interpBackwForw', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--inter4k_stepsize',  type=int, default=16, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--noPCA', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--tempbottomflowfix', action="store_true", help='NOT IMPLEMENTED YET!')
    

    # Basic Nettype
    parser.add_argument('--pcanet', action="store_true", help='Just PCA conversion!')
    parser.add_argument('--net_object', default=DCTXVFInet, choices=[DCTXVFInet], help='The type of Net')
    

    # GOOD
    parser.add_argument('--dctvfi_nf',  type=int, default=16, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--scales' ,default=[4,8,16,32,64,128], nargs='+', help='<Required> Set flag')
    parser.add_argument('--fractions' ,default=[1,4,16,64,256,1024], nargs='+', help='<Required> Set flag')
    
    parser.add_argument('--ref_feat_extrac', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--maskLess', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--imageUpInp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--allImUp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--ExacOneEV',   action="store_true", help='no features inputted at refinement step')
    #parser.add_argument('--outMaskLess',   action="store_true", help='no features inputted at refinement step')
    
    parser.add_argument('--papermodel',   action="store_true", help='no features inputted at refinement step')
        
    parser.add_argument('--validation_patch_size', type=int, default=512, help='patch size in validation')
    
    
    parser.add_argument('--meanVecParam', action="store_false", help='no features inputted at refinement step')

    # other
    parser.add_argument('--align_cornerse',action="store_true",help='no features inputted at refinement step')
    parser.add_argument('--takeBestModel',action="store_false",help='no features inputted at refinement step')
    

    # PCA adaptations 
    parser.add_argument('--oneEV', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--optimizeEV', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--noEVOptimization', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--moreTstSc', action="store_true", help='no features inputted at refinement step')


    # Hyperparameters
    parser.add_argument('--padding',  type=str, default="reflective", help='The initial learning rate')

    parser.add_argument('--XVFIPSNR',   action="store_true", help='normalize by mean vector')
    parser.add_argument('--continue_training', action='store_true', default=False, help='continue the training')
    

    # Get what you want
    parser.add_argument('--specificCheckpoint',  type=int, default=-1, help="dsadasd")


    parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
    parser.add_argument('--nf', type=int, default=64, help='base number of channels for feature maps')  # 64

    parser.add_argument('--S_trn', type=int, default=3, help='The lowest scale depth for training')
    parser.add_argument('--S_tst', type=int, default=5, help='The lowest scale depth for test')
    parser.add_argument('--timetest', action="store_true", help='My Big DCTNET')
    parser.add_argument('--testgetflowout', action="store_true", help='My Big DCTNET')
    parser.add_argument('--outMaskLess',   action="store_true", help='no features inputted at refinement step')

    return parser.parse_args()

if __name__ == '__main__':
    main()
