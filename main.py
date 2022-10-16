import argparse, os, shutil, time, random, torch, cv2, datetime, torch.utils.data, math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import cupy as cp
import scipy.fft as scF
import skimage.metrics
#from skimage.metrics import peak_signal_to_noise_ratio

from pca_comp import DCTParams,to_pca #to_dct,to_dctpca,dct_inverse
from myAdditions import ScaleIt,torch_prints,numpy_prints,MyPWC,distillation_loss

from torch.autograd import Variable
from torchvision import utils
from utils import *
from XVFInet import *
from collections import Counter
import sys
from skimage.transform import rescale
# Visualization
from torch.utils.tensorboard import SummaryWriter

def parse_args():
    desc = "PyTorch implementation for XVFI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--net_type', type=str, default='XVFInet', choices=['XVFInet'], help='The type of Net')
    parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test', 'test_custom', 'metrics_evaluation',])
    parser.add_argument('--continue_training', action='store_true', default=False, help='continue the training')

    """ Information of directories """
    superprefix = './../../../' #../../data/vimb02/
    prefix = superprefix + 'TestResults/XVFI/'

    parser.add_argument('--test_img_dir', type=str, default=prefix+'test_img_dir', help='test_img_dir path')
    parser.add_argument('--text_dir', type=str, default='./text_dir', help='text_dir path')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir', help='checkpoint_dir')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='Directory name to save training logs')

    parser.add_argument('--dataset', default='X4K1000FPS', choices=['X4K1000FPS', 'Vimeo',"Inter4K",'Xiph'],
                        help='Training/test Dataset')

    # parser.add_argument('--train_data_path', type=str, default='./X4K1000FPS/train')
    # parser.add_argument('--val_data_path', type=str, default='./X4K1000FPS/val')
    # parser.add_argument('--test_data_path', type=str, default='./X4K1000FPS/test')



    prefix = superprefix +  'X-Train/'

    parser.add_argument('--train_data_path', type=str, default=prefix+'train')
    parser.add_argument('--val_data_path', type=str, default=prefix+'val')
    parser.add_argument('--test_data_path', type=str, default=prefix+'test')


    parser.add_argument('--vimeo_data_path', type=str, default=superprefix+'vimeo_triplet') #'./vimeo_triplet'
    parser.add_argument('--hd_data_path', type=str, default=superprefix+'HDbenchmark')
    parser.add_argument('--xiph_data_path', type=str, default="XiphDataset/netflix") #'./vimeo_triplet'
    parser.add_argument('--adobe_data_path', type=str, default="Adobe240/DeepVideoDeblurring_Dataset/quantitative_datasets") #'./vimeo_triplet'

    ################################        My Hyperparameters      #############################################
    parser.add_argument('--rfb', action="store_true", help='Says if strided convolutions (from module_scale_factor) are substituted with RFB block')
    parser.add_argument('--extra_feat', action="store_true", help='If a feature extraction block is added after normal rfb')
    parser.add_argument('--no_ds_rfb', action="store_true", help='RFB no downscaling,feature extraction untouched')
    parser.add_argument('--no_ds_part_rfb', action="store_true", help='RFB no downscaling, part it')
    parser.add_argument('--give_dct_ds_in', action="store_true", help='RFB no downscaling, part it')
    parser.add_argument('--give_bilin_in', action="store_true", help='RFB no downscaling, part it')


    parser.add_argument('--validation_patch_size', type=int, default=512, help='patch size in validation')
    parser.add_argument('--test_patch_size', type=int, default=-1, help='patch size in test. If -1 no patching is done') #1024
    parser.add_argument('--pin_memory_train', action="store_true", help='faster?')
    parser.add_argument('--pin_memory_test', action="store_true", help='faster?')
    parser.add_argument('--test5scales', action="store_true", help='faster?')
    parser.add_argument('--test6scales', action="store_true", help='faster?')
    parser.add_argument('--test7scales', action="store_true", help='faster?')
    parser.add_argument('--test4scales', action="store_true", help='faster?')
    parser.add_argument('--test3scales', action="store_true", help='faster?')
    parser.add_argument('--HDupscale', action="store_true", help='faster?')


    ###########################         DCT-NET Hyperparameters         ###########################################
    # Information about model
    parser.add_argument('--parameters',  type=int, default=-1, help='My Big DCTNET') 
    

    # FORWARD WARPING & Paper general
    parser.add_argument('--softsplat', action="store_true", help='My Big DCTNET')
    parser.add_argument('--downupflow', action="store_true", help='My Big DCTNET')
    parser.add_argument('--pacupfor', action="store_true", help='My Big DCTNET')
    parser.add_argument('--ownsmooth', action="store_true", help='My Big DCTNET')
    parser.add_argument('--forwendflowloss', action="store_true", help='My Big DCTNET')
    parser.add_argument('--forrefpacflow', action="store_true", help='My Big DCTNET')
    parser.add_argument('--forrefRFBflow', action="store_true", help='My Big DCTNET')
    parser.add_argument('--complflow', action="store_true", help='My Big DCTNET')
    parser.add_argument('--complflow2', action="store_true", help='My Big DCTNET')
    parser.add_argument('--occreg', action="store_true", help='My Big DCTNET')
    parser.add_argument('--complallrefine', action="store_true", help='My Big DCTNET')
    parser.add_argument('--output_editing', action="store_true", help='My Big DCTNET')
    parser.add_argument('--outedit_lite', action="store_true", help='My Big DCTNET')
    parser.add_argument('--flowfromtto0_lite', action="store_true", help='My Big DCTNET')
    parser.add_argument('--bothflowforsynth', action="store_true", help='My Big DCTNET')
    parser.add_argument('--flowfromtto0', action="store_true", help='My Big DCTNET')
    parser.add_argument('--ownoccl', action="store_true", help='My Big DCTNET')
    parser.add_argument('--iteratpacup', action="store_true", help='My Big DCTNET')
    parser.add_argument('--bigallcomb', action="store_true", help='My Big DCTNET')
    parser.add_argument('--sminterp', action="store_true", help='My Big DCTNET')
    parser.add_argument('--sminterpWT', action="store_true", help='My Big DCTNET')
    parser.add_argument('--tparam',  type=float, default=1, help='no features inputted at refinement step')
    parser.add_argument('--minmaxinterp', action="store_true", help='My Big DCTNET')
    parser.add_argument('--minus1bwarpEnd', action="store_true", help='My Big DCTNET')
    parser.add_argument('--minus1bwarp', action="store_true", help='My Big DCTNET')
    parser.add_argument('--outlgraycorrec', action="store_true", help='My Big DCTNET')
    parser.add_argument('--noResidAddup', action="store_true", help='My Big DCTNET')
    parser.add_argument('--cutoffUnnec', action="store_true", help='My Big DCTNET')
    parser.add_argument('--fixsmoothtwistup', action="store_true", help='My Big DCTNET')
    parser.add_argument('--impmasksoftsplat', action="store_true", help='My Big DCTNET')
    parser.add_argument('--TOptimization', action="store_true", help='My Big DCTNET')
    parser.add_argument('--sminterpInpIm', action="store_true", help='My Big DCTNET')
    parser.add_argument('--tempAdamfix', action="store_true", help='My Big DCTNET')
    parser.add_argument('--simpleEVs', action="store_true", help='My Big DCTNET')
    parser.add_argument('--lightrefine', action="store_true", help='My Big DCTNET')
    parser.add_argument('--fixpacupfor', action="store_true", help='My Big DCTNET')
    parser.add_argument('--smallenrefine', action="store_true", help='My Big DCTNET')
    parser.add_argument('--interpOrigForw', action="store_true", help='My Big DCTNET')
    parser.add_argument('--interpBackwForw', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--pwcbottomLowLr', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--pwclr',  type=float, default=0.1, help='no features inputted at refinement step')
    parser.add_argument('--dsstart',  type=int, default=1, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--resbefaf',  type=int, default=1, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--justpwcflow', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--distilflowloss', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--smoothimages', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--lowresvers', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--justpwcadaption', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--pwcresid', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--flowrefinenorm', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--teacherflowresid', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--inter4k_stepsize',  type=int, default=16, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--teachparam',  type=float, default=1, help='no features inputted at refinement step')
    parser.add_argument('--distillparam',  type=float, default=1, help='no features inputted at refinement step')
    parser.add_argument('--pwcflowrefine',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--pwcbottomflow', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--simpleupsam', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--additivePWC', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--addPWCOneFlow', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--noPCA', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--tempbottomflowfix', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--pixtransupsam', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--oldpacupfor', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--nonadditivepac', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--raftupflowfeat', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--raftupflowimage', action="store_true", help='NOT IMPLEMENTED YET!')
    parser.add_argument('--raftlevel0', action="store_true", help='NOT IMPLEMENTED YET!')
    
    # Get what u want
    parser.add_argument('--timetest', action="store_true", help='My Big DCTNET')
    parser.add_argument('--xiph2k', action="store_true", help='My Big DCTNET')
    parser.add_argument('--xtest2k', action="store_true", help='My Big DCTNET')
    parser.add_argument('--stoptestat',  type=int, default=-1, help='number of feature maps put into Net per imagechannel')
    # Manipulation
    parser.add_argument('--testgetflowout', action="store_true", help='My Big DCTNET')
    parser.add_argument('--temptestimages', action="store_true", help='My Big DCTNET')
    parser.add_argument('--jumptotest',  type=int, default=-1, help='number of feature maps put into Net per imagechannel')

    # Basic Nettype
    parser.add_argument('--dctnet', action="store_true", help='My Big DCTNET') 
    parser.add_argument('--pcanet', action="store_true", help='Just PCA conversion!')
    parser.add_argument('--net_object', default=DCTXVFInet, choices=[XVFInet,DCTXVFInet], help='The type of Net')
    
    # Structural Changes OLD
    parser.add_argument('--no_refine', action="store_true", help='no refinement in the end, just warping')
    parser.add_argument('--norm_image_warp', action="store_true", help='warp the normal image, not the DCT version')
    parser.add_argument('--feat_input', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--advanced_feature_extraction', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--vgg_feat_input', action="store_true", help='no features inputted at refinement step')
    
    # GOOD
    parser.add_argument('--ds_normInput', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--normImageScale', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--zeroOwnPar', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--dctvfi_nf',  type=int, default=16, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--scales' ,default=[4,8,16,32,64,128], nargs='+', help='<Required> Set flag')
    parser.add_argument('--fractions' ,default=[1,4,16,64,256,1024], nargs='+', help='<Required> Set flag')
    
    parser.add_argument('--ref_feat_extrac', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--pixelshuffle', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--archTest', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--archTest2', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--maskLess', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--ignoreFeatx', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--littlePCAExtra', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--pacFlow',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--pacFlow2',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--pacFlow3',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--supLight',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--zeroUpsamp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--zeroUpsamp2',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--cannyInput',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--imageUpInp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--doubleParameters',  type=float, default=1, help='no features inputted at refinement step')
    parser.add_argument('--allImUp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--ExacOneEV',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--extra4Layer',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--addedFlow',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--outMaskLess',   action="store_true", help='no features inputted at refinement step')
    
    
    # Upsampling
    parser.add_argument('--pacUpsamplingNew', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--upsamplingLayers', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--pacUpsampling', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--upsamplingEpoch',  type=int, default=-1, help='no features inputted at refinement step')
    parser.add_argument('--addUpsam', action="store_true", help='no features inputted at refinement step')
    
    
    # temp fuck
    parser.add_argument('--meanVecParam', action="store_false", help='no features inputted at refinement step')

    # other
    parser.add_argument('--align_cornerse',action="store_true",help='no features inputted at refinement step')
    parser.add_argument('--takeBestModel',action="store_false",help='no features inputted at refinement step')
    parser.add_argument('--testmessage', type=str, default="",help='no features inputted at refinement step')
    

    # Losses
    parser.add_argument('--warping_loss', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--warp_alpha',  type=float, default=0.5, help='The initial learning rate')
    parser.add_argument('--endflowwarploss',  action="store_true", help='The initial learning rate')
    parser.add_argument('--eFWLEpoch',  type=int, default=80, help='The initial learning rate')
    parser.add_argument('--orthLoss',  action="store_true", help='The initial learning rate')
    parser.add_argument('--evlr',  type=float, default=0.1, help='The initial learning rate')
    
    # PCA adaptations 
    parser.add_argument('--oneEV', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--optimizeEV', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--noEVOptimization', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--moreTstSc', action="store_true", help='no features inputted at refinement step')

    # Normalization
    parser.add_argument('--mean_vector_norm',   action="store_true", help='normalize by mean vector')
    parser.add_argument('--maxmin_vec',   action="store_true", help='normalize by mean vector')
    parser.add_argument('--weightMat',   action="store_true", help='normalize by mean vector')

    # Hyperparameters
    parser.add_argument('--smoothness',  type=float, default=0.5, help='The initial learning rate')
    parser.add_argument('--padding',  type=str, default="reflective", help='The initial learning rate')
    parser.add_argument('--flow_padding',  type=str, default="constant", help='The initial learning rate')

    parser.add_argument('--XVFIPSNR',   action="store_true", help='normalize by mean vector')
    
    # Automatically set
    parser.add_argument('--one_layer', action="store_true", help='no features inputted at refinement step')
    
    # Get what you want
    parser.add_argument('--seedbyexp', action="store_true")
    parser.add_argument('--samefirstpic', action="store_true")    
    parser.add_argument('--directly_save_model', action="store_true")
    parser.add_argument('--flowtest', action="store_true")
    parser.add_argument('--no_validation', action="store_true")
    parser.add_argument('--testsets', default=["Inter4K88","Inter4K816",'X4K1000FPS','Xiph'] ,nargs='+', )
    parser.add_argument('--specificCheckpoint',  type=int, default=-1, help="dsadasd")

         
    """ Hyperparameters for Training (when [phase=='train']) """
    parser.add_argument('--epochs', type=int, default=200, help='The number of epochs to run')
    parser.add_argument('--freq_display', type=int, default=100, help='The number of iterations frequency for display')
    parser.add_argument('--save_img_num', type=int, default=4,
                        help='The number of saved image while training for visualization. It should smaller than the batch_size')
    parser.add_argument('--init_lr', type=float, default=1e-4, help='The initial learning rate')
    parser.add_argument('--lr_dec_fac', type=float, default=0.25, help='step - lr_decreasing_factor')
    parser.add_argument('--lr_milestones', default=[100, 150, 180],nargs='+',)
    parser.add_argument('--lr_dec_start', type=int, default=0,
                        help='When scheduler is StepLR, lr decreases from epoch at lr_dec_start')
    parser.add_argument('--batch_size', type=int, default=8, help='The size of batch size.')
    parser.add_argument('--weight_decay', type=float, default=0, help='for optim., weight decay (default: 0)')

    parser.add_argument('--need_patch', default=True, help='get patch form image while training')
    parser.add_argument('--img_ch', type=int, default=3, help='base number of channels for image')
    parser.add_argument('--nf', type=int, default=64, help='base number of channels for feature maps')  # 64
    parser.add_argument('--module_scale_factor', type=int, default=4, help='sptial reduction for pixelshuffle')
    parser.add_argument('--patch_size', type=int, default=384, help='patch size')
    parser.add_argument('--num_thrds', type=int, default=8, help='number of threads for data loading')
    parser.add_argument('--loss_type', default='L1', choices=['L1', 'MSE', 'L1_Charbonnier_loss'], help='Loss type')

    parser.add_argument('--S_trn', type=int, default=3, help='The lowest scale depth for training')
    parser.add_argument('--S_tst', type=int, default=5, help='The lowest scale depth for test')

    """ Weighting Parameters Lambda for Losses (when [phase=='train']) """
    parser.add_argument('--rec_lambda', type=float, default=1.0, help='Lambda for Reconstruction Loss')

    """ Settings for Testing (when [phase=='test' or 'test_custom']) """
    parser.add_argument('--saving_flow_flag', default=False)
    parser.add_argument('--multiple', type=int, default=8, help='Due to the indexing problem of the file names, we recommend to use the power of 2. (e.g. 2, 4, 8, 16 ...). CAUTION : For the provided X-TEST, multiple should be one of [2, 4, 8, 16, 32].')
    parser.add_argument('--metrics_types', type=list, default=["PSNR", "SSIM", "tOF"], choices=["PSNR", "SSIM", "tOF"])

    """ Settings for test_custom (when [phase=='test_custom']) """
    parser.add_argument('--custom_path', type=str, default='./custom_path', help='path for custom video containing frames')

    parser.add_argument('--halfXVFI', action="store_true", help='no features inputted at refinement step')
    

    return check_args(parser.parse_args())


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --text_dir
    check_folder(args.text_dir)

    # --log_dir
    check_folder(args.log_dir)

    # --test_img_dir
    check_folder(args.test_img_dir)

    return args

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    args = parse_args()
    # SEED
    myseed = 100
    if args.seedbyexp:
        myseed = args.exp_num
    torch.manual_seed(myseed)
    genseed = torch.Generator()
    genseed.manual_seed(myseed)
    import random
    random.seed(myseed)
    np.random.seed(myseed)
    #torch.use_deterministic_algorithms(True)
    #torch.autograd.set_detect_anomaly(True)
    # Just a test routine
    if(args.test7scales):
        args.fractions = [4, 16, 64, 256, 1024,4096,16384,65_536]
        args.scales = [8,16,32,64,128,256,512,1024]
        args.moreTstSc = True
        args.phase = "test"
        args.S_tst = 7
    if(args.test6scales):
        args.fractions = [4, 16, 64, 256, 1024,4096,16384]
        args.scales = [8,16,32,64,128,256,512]
        args.moreTstSc = True
        args.phase = "test"
        args.S_tst = 6
    if(args.test5scales):
        args.fractions = [4, 16, 64, 256, 1024,4096]
        args.scales = [8,16,32,64,128,256]
        args.moreTstSc = True
        args.phase = "test"
        args.S_tst = 5
    if(args.test4scales):
        args.fractions = [4, 16, 64, 256, 1024]
        args.scales = [8,16,32,64,128]
        args.moreTstSc = True
        args.phase = "test"
        args.S_tst = 4
    if(args.test3scales):
        args.phase = "test"
        
    args.fractions = [int(i) for i in args.fractions]
    args.scales = [int(i) for i in args.scales]
    args.dctvfi_nf = args.scales[0]**2//args.fractions[0]
    args.padding = "reflect" if(args.pcanet or args.dctnet)else "constant"

    if(len(args.scales) != len(args.fractions)):
        raise Exception("Scales and Fractions array don't have the same length!")
    if(args.no_ds_part_rfb):
        if(args.S_trn != 3 or args.S_tst != 3):
            print("dude you can part it only in three parts!!!")
            sys.exit()
    if(not args.dctnet and not args.pcanet):
        args.net_object = XVFInet
    if(args.flowtest):
        args.continue_training = True

    if(args.S_trn == 0):
        args.one_layer = True
    if(args.maxmin_vec and args.mean_vector_norm):
        sys.exit()
    # if(args.patch_size == 384 and args.dctnet):
    #    args.patch_size = 768
    if args.dataset != 'X4K1000FPS':
        args.multiple = 2
    if args.dataset == 'Vimeo':
        if args.phase != 'test_custom':
            args.multiple = 2
        if(not args.pcanet):
            args.S_trn = 1
            args.S_tst = 1
        args.module_scale_factor = 2
        args.patch_size = 256
        args.batch_size = 16
        args.eFWLEpoch = 30
        print('vimeo triplet data dir : ', args.vimeo_data_path)

    #assert not args.noResidAddup or args.sminterp

    assert not(args.archTest and args.pacFlow)
    assert not(args.archTest and args.pacFlow2)
    assert not(args.archTest and args.pacFlow3)
    assert not args.ExacOneEV or args.allImUp 
    assert  (args.imageUpInp == (not args.ExacOneEV )) or (not args.imageUpInp) and not args.ExacOneEV

    if(args.pcanet):
        assert args.S_trn == args.S_tst or args.moreTstSc
        args.takeBestModel = True
    else:
        args.takeBestModel = False
    assert args.zeroUpsamp == args.ignoreFeatx
    

    print("Exp:", args.exp_num)
    args.model_dir = args.net_type + '_' + args.dataset + '_exp' + str(
        args.exp_num)  # ex) model_dir = "XVFInet_X4K1000FPS_exp1"

    if args is None:
        exit()
    for arg in vars(args):
        print('# {} : {}'.format(arg, getattr(args, arg)))
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')  # will be used as "x.to(device)"
    torch.cuda.set_device(device)  # change allocation of current GPU
    # caution!!!! if not "torch.cuda.set_device()":
    # RuntimeError: grid_sampler(): expected input and grid to be on same device, but input is on cuda:1 and grid is on cuda:0
    print('Available devices: ', torch.cuda.device_count())
    print('Current cuda device: ', torch.cuda.current_device())
    print('Current cuda device name: ', torch.cuda.get_device_name(device))
    if args.gpu is not None:
        print("Use GPU: {} is used".format(args.gpu))

    

    """ Initialize a model """
    model_net = args.net_object(args).apply(weights_init).to(device) # Here the model is loaded from XVFI-net and applies the args
    criterion = [set_rec_loss(args).to(device),set_smoothness_loss_forward(args=args).to(device) if(args.ownsmooth)else set_smoothness_loss().to(device),set_warping_loss(args).to(device),set_warping_loss_endflow(args).to(device)]

    # print("model net up")
    # time.sleep(3)

    # Parameter Print
    print("Total Parameters:           ",sum(p.numel() for p in model_net.parameters()))
    print("Total learnable Parameters: ",sum(p.numel() for p in model_net.parameters() if p.requires_grad))
    args.parameters = sum(p.numel() for p in model_net.parameters() if p.requires_grad)
    SM = save_manager(args)
    # to enable the inbuilt cudnn auto-tuner
    # to find the best algorithm to use for your hardware.
    cudnn.benchmark = True

    # Inter4K_Train(args)
    # sys.exit()
    if args.phase == "train":
        train(model_net, criterion, device, SM, args,genseed)
        epoch = args.epochs - 1

    elif args.phase == "test" or args.phase == "metrics_evaluation" or args.phase == 'test_custom' or args.phase=="train":
        # Fetches LATEST Model
        checkpoint = SM.load_model(takeBestModel=args.takeBestModel,specific=args.specificCheckpoint)
        SM.write_info("New Test has started: "+ args.testmessage)
        model_net.load_state_dict(checkpoint['state_dict_Model'])
        if(model_net.args.oneEV):
            if(not args.meanVecParam):
                model_net.pick_norm_vec(checkpoint['used_pcas'])
            model_net.save_params(checkpoint['paramsPCA'])
        epoch = checkpoint['last_epoch']

    # epoch = 1
    postfix = '_final_x' + str(args.multiple) + '_S_tst' + str(args.S_tst)
    if args.phase != "test":
        sys.exit()
    
    if args.phase != "metrics_evaluation":
        print("\n-------------------------------------- Final Test starts -------------------------------------- ")
        print('Evaluate on test set (final test) with multiple = %d ' % (args.multiple))

        for i in args.testsets:
            args.dataset = i
            temMultiple = {"X4K1000FPS": 8,"XTest2KC":8,"Inter4K88":8,"Inter4K816":8,"Xiph": 2,"Xiph2KC":2,"Vimeo":2 ,"Adobe240": 8,"HD":4}
            final_test_loader = get_test_data(args, args.dataset,multiple=temMultiple[i],
                                              validation=False,specific=i)  # multiple is only used for X4K1000FPS

            testLoss, testPSNR, testSSIM, final_pred_save_path, PSNRsList = test(final_test_loader, model_net,
                                                                      criterion, epoch,
                                                                      args, device,
                                                                      multiple=temMultiple[i],
                                                                      postfix=postfix, validation=False)
            SM.write_info('Final 4k frames PSNR '+i + ' : {:.4}\n'.format(testPSNR))
            print('Final 4k frames PSNR '+i + ' : {:.4}\n'.format(testPSNR))
            if(args.dataset == "Inter4K88" or args.dataset == "Inter4K816"):
	            printstring = " ".join([str(index)+": "+str(ttime.avg)+ " || " for index,ttime in enumerate(PSNRsList)])
	            print(printstring)
	            SM.write_info(printstring)

    if args.dataset == 'X4K1000FPS' and args.phase != 'test_custom':
        final_pred_save_path = os.path.join(args.test_img_dir, args.model_dir, 'epoch_' + str(epoch).zfill(5)) + postfix
        metrics_evaluation_X_Test(final_pred_save_path, args.test_data_path, args.metrics_types,
                                  flow_flag=args.saving_flow_flag, multiple=args.multiple)



    print("------------------------- Test has been ended. -------------------------\n")
    print("Exp: ", args.exp_num)

# (x+1)/2 is done here to make it [0,1] before transforming, BUT NOT WHEN ONLYPCA is TRUE
def preprocessing(args,input_frames,frameTList,device,frameT,train=False,onlyPCA=False,model_net=None):

    shap = input_frames.shape
    input_frames = input_frames.reshape(shap[0],-1,shap[3],shap[4])
    frameT_adap = frameT

    shap = input_frames.shape
    H,W = shap[2:4]
    data_used = 0.5 if(args.phase=="train")else 0.01

    params = model_net.params#[]
    # if(params == None and False):
    #     #raise Exception("Params are not None, it should not get here!")
    #     params = []
    #     for i in range(args.S_trn+1):
    #         params.append(DCTParams(wiS=args.scales[i],components_fraction=1/args.fractions[i],data_used=data_used ) ) 
    #     for i in params:
    #         i.weightMat = torch.load("weightMatrices/wTensor"+str(i.wiS)+"x"+str(i.wiS)+".pt").to(device)
    #         i.weightMat.required_grad = False

    # all_pcas = []
    # for l in range(len(params)):
    #     all_pcas.append([0 for i in range(shap[0])])
    

    # TODO: doppelerstellung hier, uebergebe input_gpu bei to_dctpca stattdessen
    # Input_gpu_big has all components as that is needed when warping, input_gpu_small on the other hand only uses 1/8 of it
    diff_scales = len(args.scales) - len(params)
    for i in range(diff_scales ):
    	params.append(DCTParams(wiS=8,components_fraction=1/4,data_used=data_used ) )
    input_gpuList = []
    for l in range(len(params)):
    	input_gpuList.append( torch.zeros((shap[0],int(args.img_ch*2*(params[l].wiS**2)*params[l].components_fraction) ,H//params[l].wiS,W//params[l].wiS),device=device))


		
    # for l in range(len(all_pcas)):
    #     if(args.oneEV):
    #         pass #pca = model_net.used_pcas[l]    #to_pca(input_frames[0,:,:,:],params[l],components_fraction=0,args=args)
    #     else:
    #         pca = 0

    #     for i in range(shap[0]):
    #     # DCTPCA/PCA
    #         if(not args.oneEV):
    #             if(onlyPCA):
    #                 temp,tempPca = to_pca(input_frames[i,:,:,:],params[l],components_fraction=0,args=args,pca=pca) if(not args.oneEV)else None,None # take only 1/8 of the components
    #                 all_pcas[l][i] = pca if(args.oneEV) else tempPca
    #             else:
    #                 temp,all_pcas[l][i] = to_dctpca(input_frames[i,:,:,:],params[l],components_fraction=0) # take only 1/8 of the components
    #             if(not args.oneEV):
    #                 input_gpuList[l][i,:] = torch.as_tensor(temp,device=device)

    #         if(train and (not onlyPCA)):
    #             # DCT
    #             frameTs[l][i,:] = torch.from_numpy(to_dct((frameT_adap[i,:].numpy()+1)/2,wiS=params[l].wiS))
                   

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


    return input_gpuList

# Input_frames [C,T,H,W] is between [-1,1]
def once_preprocessed(input_frames,model_net,args,device):
    H,W = input_frames.shape[2:4]
    data_used = 0.5 if(args.phase=="train")else 0.01
    params = []
    for i in range(len(args.scales)):#range(args.S_tst+1):
        if(i == 0 and args.imageUpInp):
            params.append(DCTParams(wiS=8,components_fraction=1/4,data_used=data_used ) )
        elif(args.allImUp):
            temFrac = args.fractions[args.scales.index(8)]
            params.append(DCTParams(wiS=8,components_fraction=1/temFrac,data_used=data_used ) )
        else:
            params.append(DCTParams(wiS=args.scales[i],components_fraction=1/args.fractions[i],data_used=data_used ) )

    print(params)
    if(args.weightMat):
        for i in params:
            i.weightMat = torch.load("weightMatrices/wTensor"+str(i.wiS)+"x"+str(i.wiS)+".pt").to(device)
            i.weightMat.required_grad = False
    model_net.save_params(params)
    # for i in params:
    #     print(i.weightMat.get_device())
    all_pcas = []


    if(args.simpleEVs):
        if(args.dsstart > 1):
            _,all_pcas = to_pca(F.interpolate(input_frames,scale_factor=1/args.dsstart,mode="bilinear").permute(1,0,2,3).reshape(-1,H,W),params[0],components_fraction=0,args=args)
        else:
            _,all_pcas = to_pca(input_frames.permute(1,0,2,3).reshape(-1,H,W),params[0],components_fraction=0,args=args)
    else:
        for index,i in enumerate(params):
            if(index == 0 and args.imageUpInp):
                #print("input frames: ",input_frames.shape)
                tempInp =  F.interpolate(input_frames,scale_factor=2,mode="nearest").permute(1,0,2,3).reshape(-1,H*2,W*2)    
                #print("temp inp: ",tempInp.shape)
                _,pca = to_pca(tempInp,params[0],components_fraction=0,args=args)
            elif(args.allImUp and args.scales[index] != 8):
                tempMul = 8/args.scales[index]
                tempInp =  F.interpolate(input_frames,scale_factor=8/args.scales[index],mode="nearest").permute(1,0,2,3).reshape(-1,int(H*tempMul),int(W*tempMul))
                _,pca = to_pca(tempInp,params[index],components_fraction=0,args=args)
            else:
                _,pca = to_pca(input_frames.permute(1,0,2,3).reshape(-1,H,W),i,components_fraction=0,args=args)
            all_pcas.append(pca)



    model_net.pick_pca(all_pcas)
    del all_pcas
    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()
  

def train(model_net, criterion, device, save_manager, args,genseed):

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    SM = save_manager
    tbWriter = SummaryWriter(log_dir="runs/"+str(args.exp_num))
    stepper = 0
    reconStepper = 0
    valTB_tracker = {"stepper": 0,"tbWriter":tbWriter}

    multi_scale_recon_loss = criterion[0]
    smoothness_loss = criterion[1]
    warping_rec_loss = criterion[2]
    warping_endflow_loss = criterion[3]

    #print(model_net.ev_params)
    if(args.pwcbottomLowLr):
        optimIn = [{"params": model_net.ev_params, "lr": args.init_lr*args.evlr}
                    ,{"params": model_net.base_modules.parameters()}
                    ,{"params": model_net.mypwc.flow_predictor.parameters(), "lr": args.init_lr*args.pwclr}]
    else:
        optimIn = [{"params": model_net.ev_params, "lr": args.init_lr*args.evlr}
                ,{"params": model_net.base_modules.parameters()}]
    if(args.noEVOptimization):
        optimIn = [{"params": model_net.base_modules.parameters()}]
    optimizer = optim.Adam(optimIn, lr=args.init_lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)  # optimizer

    #print([i["lr"] for i in optimizer.param_groups])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_dec_fac)

    last_epoch = 0
    best_PSNR = 0.0

    # Here ---LATEST----- is loaded
    if args.continue_training:
        if(args.TOptimization):
            checkpoint = SM.load_model(takeBestModel=True)
        else:
            checkpoint = SM.load_model()
        last_epoch = checkpoint['last_epoch'] + 1
        best_PSNR = checkpoint['best_PSNR']
        model_net.load_state_dict(checkpoint['state_dict_Model'])
        if(args.tempAdamfix):
        	for i in checkpoint["state_dict_Optimizer"]["state"].keys():
        		checkpoint["state_dict_Optimizer"]["state"][i]["step"] = checkpoint["state_dict_Optimizer"]["state"][i]["step"].to("cpu")
        if(not args.TOptimization):
            optimizer.load_state_dict(checkpoint['state_dict_Optimizer'])
            scheduler.load_state_dict(checkpoint['state_dict_Scheduler'])
        if(model_net.args.oneEV and not args.meanVecParam):
            #print(checkpoint)
            model_net.pick_norm_vec(checkpoint['used_pcas'])
        model_net.save_params(checkpoint['paramsPCA'])
        print("Optimizer and Scheduler have been reloaded. ")
        # print(checkpoint["state_dict_Optimizer"]["state"].keys())
        # print(checkpoint["state_dict_Optimizer"]["state"][53])


    # for p in optimizer.param_groups:
    #     print(p["params"])

    scheduler.milestones = Counter(args.lr_milestones)
    scheduler.gamma = args.lr_dec_fac
    print("scheduler.milestones : {}, scheduler.gamma : {}".format(scheduler.milestones, scheduler.gamma))
    if(args.flowtest):
        start_epoch = 0#last_epoch
    else:
        start_epoch = last_epoch


    #torch.autograd.set_detect_anomaly(True)

    # switch to train mode
    model_net.train()
    model_net.to(device)
    #print("BIG PRINT: ",model_net.params,model_net.used_pcas)
    start_time = time.time()
    if(not args.continue_training):
        SM.write_info('Epoch\ttrainLoss\tWarpLoss\tValLoss\ttestPSNR\tbest_PSNR\n')
    print("[*] Training starts")

    # Main training loop for total epochs (start from 'epoch=0')
    valid_loader = get_test_data(args,"X4K1000FPS" if(args.dataset=="Inter4K88" or args.dataset=="Inter4K816")else args.dataset, multiple=4, validation=True)  # multiple is only used for X4K1000FPS
    
    tempIndex = 0
    if(args.TOptimization):
        for param in model_net.ev_params:
            param.requires_grad = False
        for index,param in enumerate(model_net.base_modules.named_parameters()):
            #print(param[0])
            if(param[0] != "1.T_param"):
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True
            
    
    if(args.distilflowloss):
        pwcnet = MyPWC(args)
     

    once = True
    first_frame = None
    temploader,_ = get_train_data(args,
                                      max_t_step_size=32,device=device,shuffle=not (args.seedbyexp or args.samefirstpic))  # max_t_step_size (temporal distance) is only used for X4K1000FPS
    for index,(fr,_) in enumerate(temploader):
        if index != args.exp_num % 10 and not args.seedbyexp and not args.samefirstpic:
            continue
        tempframe = fr[:, :, :-1, :]
        first_frame = tempframe[0,:,:,:,:].detach()
        print(" IAAAAAAAAAAAAAAAAA MMMMM")
        first_frame_np =np.around(denorm255_np(np.transpose(first_frame[:,0,:].numpy(), [1, 2, 0])))
        nameaddup = "ablation" if(args.samefirstpic) else ""
        cv2.imwrite("tempTest/"+str(args.exp_num)+nameaddup+".png",first_frame_np)
        break

    for epoch in range(start_epoch, args.epochs):
        train_loader,_ = get_train_data(args,
                                      max_t_step_size=32,device=device,genseed=None)  # max_t_step_size (temporal distance) is only used for X4K1000FPS

        batch_time = AverageClass('batch_time[s]:', ':6.3f')
        losses = AverageClass('Loss:', ':.4e')
        warp_loss = AverageClass('Warp:',':.4e')
        progress = ProgressMeter(len(train_loader), batch_time, losses, prefix="Epoch: [{}]".format(epoch))

        print('Start epoch {} at [{:s}], learning rate : [{}]'.format(epoch, (str(datetime.now())[:-7]),
                                                                      optimizer.param_groups[0]['lr']))
        print("Learning rates: ", [i["lr"] for i  in optimizer.param_groups])

        start_time_epoch = time.time()
        # train for one epoch
        for trainIndex, (frames, t_value) in enumerate(train_loader):



            
            
            input_frames = frames[:, :, :-1, :] # [B, C, T, H, W]  # 8,3,2,384,384
            B, C, T, H, W = input_frames.shape
            frameT = frames[:, :, -1, :]  # [B, C, H, W]     # 8,3,384,384
            frameTList = []

            if(once  and not args.continue_training and args.pcanet):
                once_preprocessed(first_frame,model_net,args,device)
            #print("Total Parameters: ",sum(p.numel() for p in model_net.parameters() if p.requires_grad ))

            #break
            # VIELLEICHT EINFACH DCT VORHER IM DATALOADER MACHEN!
            #[chan * components,blocks_y,blocks_x]
            if(args.pcanet):
                input_gpuList = preprocessing(args,input_frames,frameTList,device,frameT,train=True,onlyPCA=True,model_net=model_net)
                if(args.ds_normInput):
                    # 1 1/2  1/4  1/8
                    if(args.smoothimages):
                        input_gpu = [F.interpolate(model_net.vfinet.get_gaussian_kernel(sigma=(( args.scales[i]/args.scales[0])-1)/2)(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W)), scale_factor=args.scales[0]/ args.scales[i],mode='bicubic', 
                            align_corners=args.align_cornerse).to(device).reshape(B,T,C,int(H*(args.scales[0]/ args.scales[i])),int(W*(args.scales[0]/ args.scales[i]))).permute(0,2,1,3,4) if(i!=0) else input_frames.to(device)  for i in range(args.S_trn+1)]
                    else:
                        input_gpu = [F.interpolate(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W), scale_factor=args.scales[0]/ args.scales[i],mode='bicubic', 
                            align_corners=args.align_cornerse).to(device).reshape(B,T,C,int(H*(args.scales[0]/ args.scales[i])),int(W*(args.scales[0]/ args.scales[i]))).permute(0,2,1,3,4) if(i!=0) else input_frames.to(device)  for i in range(args.S_trn+1)]#print(torch.mean(input_frames[1,:,0,:,:].clone()))
                    #cv2.imwrite("tempTest/biggestPic.png",(input_gpu[0][1,:,0,:,:].clone().permute(1,2,0).cpu().numpy()+1)/2*255)
                    #print("saved")
                else:
                    input_gpu = (input_frames.to(device)) # as to_dctpca already brings it on GPU
                    input_gpu = [input_gpu for i in range(args.S_trn+1)]

                frameT = frameT.to(device)  # ground truth for frameT
                frameTList = []
                for i in range(args.S_trn+1):
                    frameTList.append(frameT)

            elif(args.dctnet and (not args.norm_image_warp)):
                input_gpuList,params = preprocessing(args,input_frames,frameTList,device,frameT,train=True)

            elif(args.norm_image_warp):
                input_gpuList,all_pcas,params = preprocessing(args,input_frames,frameTList,device,frameT,train=True)
                input_gpu = (input_frames.to(device)) # as to_dctpca already brings it on GPU
                frameT = frameT.to(device)  # ground truth for frameT
                frameTList = []
                for i in range(args.S_trn+1):
                    frameTList.append(frameT)

            else:
                input_gpu = input_frames.to(device) # as to_dctpca already brings it on GPU
                frameT = frameT.to(device)  # ground truth for frameT
                for i in range(args.S_trn+1):
                    frameTList.append(frameT)
                
            t_value = t_value.to(device)  # [B,1]
            optimizer.zero_grad()


            #################################       MODEL FORWARD          ##################################################
            if(args.pcanet):
                pred_frameT_pyramid, pred_flow_pyramid,unref_flow_pyramid, occ_map, simple_mean, endflow, refine_out = model_net(input_gpuList, t_value,normInput=input_gpu,epoch=epoch,frameT=frameT)
            elif(args.dctnet and (not args.norm_image_warp)):    ## DCTNET
                pred_frameT_pyramid, pred_flow_pyramid,unref_flow_pyramid, occ_map, simple_mean, endflow = model_net(input_gpuList,all_pcas,params, t_value)
                # pred_frameT_pyramid DCTOnly, with fewer featuredimensions per scale level
            elif(args.norm_image_warp):
                pred_frameT_pyramid, pred_flow_pyramid,unref_flow_pyramid, occ_map, simple_mean, endflow = model_net(input_gpuList,all_pcas,params, t_value,normInput=input_gpu)
            else:               ## XVFINET
                pred_frameT_pyramid, pred_flow_pyramid, occ_map, simple_mean, endflow = model_net(input_gpu, t_value)


            #####################################     LOSSES      ####################################
            rec_loss = 0.0
            smooth_loss = 0.0
            flow_distil = torch.tensor(0.0,device=device)
            warping_loss = torch.tensor(0.0,device=device)
            orthLoss = torch.tensor(0.0,device=device)
            teachloss = torch.tensor(0.0,device=device)
            
            for l, pred_frameT_l in enumerate(pred_frameT_pyramid):
                temp = 1/ (2**l)
                #torch_prints(pred_frameT_l,"pred_frameT_l")

                if((args.no_ds_rfb or args.dctnet or args.pcanet) and not args.ds_normInput):
                    temp = 0
                elif(args.pcanet):
                    temp = args.scales[0]/args.scales[l]
                    if(l != 0):
                    	temp/=args.dsstart 
                #print(pred_frameT_l.shape,frameTList[l].shape)
                if(args.TOptimization):
                    tempLoss = torch.mean((pred_frameT_l - F.interpolate(frameTList[l], scale_factor=temp, 
                                                                                       mode='bicubic', align_corners=args.align_cornerse))**2)
                else:
                    tempLoss = multi_scale_recon_loss(pred_frameT_l,F.interpolate(frameTList[l], scale_factor=temp, 
                                                                                   mode='bicubic', align_corners=args.align_cornerse))
                rec_loss += args.rec_lambda * tempLoss
                tbWriter.add_scalar(tag="Recon Loss Scale"+str(args.scales[l]),scalar_value=tempLoss.item(),global_step=reconStepper)
                reconStepper += 1

            # if(args.dctnet or args.pcanet):
            #     #args.module_scale_factor = 8
            #     pred_flow_pyramid[0] = pred_flow_pyramid[0][:,:,:args.patch_size//args.scales[0],:args.patch_size//args.scales[0]]
            #     unref_flow_pyramid[0] = unref_flow_pyramid[0][:,:,:args.patch_size//args.scales[0],:args.patch_size//args.scales[0]]

            # schwierig schau ich noooooooooooch
            if(args.ownsmooth):
                if(args.pacupfor or args.oldpacupfor or args.nonadditivepac):
                    #print(pred_flow_pyramid[0].shape,input_frames[:,:,0,:].shape)
                    smooth_loss += args.smoothness * smoothness_loss(pred_flow_pyramid[0],
                                                         input_frames[:,:,0,:].to(device),input_frames[:,:,1,:].to(device))
                elif(args.lowresvers):
                    smooth_loss += args.smoothness * smoothness_loss(pred_flow_pyramid[0],
                                                 F.interpolate(input_frames[:,:,0,:].to(device), scale_factor=1 / 2,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse),F.interpolate(input_frames[:,:,1,:].to(device), scale_factor=1 / 2,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse))
                else:
                    smooth_loss += args.smoothness * smoothness_loss(pred_flow_pyramid[0],
                                                 F.interpolate(input_frames[:,:,0,:].to(device), scale_factor=1 / (args.dsstart *args.scales[0]) if(args.pcanet)else 1/args.module_scale_factor,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse),F.interpolate(input_frames[:,:,1,:].to(device), scale_factor=1 / (args.dsstart *args.scales[0]) if(args.pcanet)else 1/args.module_scale_factor,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse))
            else:
            	smooth_loss += args.smoothness * smoothness_loss(pred_flow_pyramid[0],
                                                 F.interpolate(frameTList[0], scale_factor=1 / args.scales[0] if(args.pcanet)else 1/args.module_scale_factor,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse))  # Apply 1st order edge-aware smoothness loss to the fineset level
            
            if(args.forwendflowloss):
                endflowforloss = set_warping_loss_endflow_forward()
                warping_loss += args.warp_alpha * endflowforloss(endflow[0],input_frames[:,:,0,:].to(device),input_frames[:,:,1,:].to(device),t_value)
            if(args.warping_loss):
                fine_unrefined_flow = args.scales[0] * F.interpolate(unref_flow_pyramid[0], scale_factor=args.scales[0], mode='bicubic',align_corners=args.align_cornerse)
                epochAlpha = args.warp_alpha * (1 - torch.exp(-torch.tensor((args.epochs/4-epoch),device=device))) if(epoch <= args.epochs/4)else 0
                warping_loss += epochAlpha * warping_rec_loss(input_frames.to(device),fine_unrefined_flow)
                
            if(args.endflowwarploss):# 384,96,48
                for index,i in enumerate(endflow):
                    in_temp = F.interpolate(frameT, scale_factor=args.scales[0]/ args.scales[index],mode='bilinear',align_corners=args.align_cornerse) if(index>0)else frameT
                    warping_loss += args.warp_alpha * ((1- epoch/(args.eFWLEpoch*1.1))**2  * (warping_endflow_loss(input_gpu[index],in_temp,i)) if(epoch <args.eFWLEpoch) else 0)
                #endflow endflow ist liste (level) von listen (flow1,flow2)
            
            input_gpu = 0
            input_gpuList = 0
            in_temp = 0
            torch.cuda.empty_cache()
            if(args.distilflowloss):
            	with torch.no_grad():
                	gtflow = pwcnet.get_flow(input_frames[:,:,0,:].to(device),input_frames[:,:,1,:].to(device))
                	#[flow10,flow01]
            	flow_distil += distillation_loss(unref_flow_pyramid,gtflow,device)

            if(args.orthLoss):
                for index,evs in enumerate(model_net.EVs):
                    if(index == len(args.scales)):
                        continue
                    for kev in range(args.dctvfi_nf):
                        for lev in range(args.dctvfi_nf):
                            #print("index: ",lev,kev)
                            if(kev == lev or evs == None):
                                continue
                            orthLoss += torch.sum(evs[kev,:] * evs[lev,:])
                assert len(orthLoss.shape) == 0
                assert not orthLoss.isnan(), "orthLoss is nan"
            ###########################         Backward and Saving         ####################################################

            rec_loss /= len(pred_frameT_pyramid)
            pred_frameT = pred_frameT_pyramid[0]  # final result I^0_t at original scale (s=0)

            if(args.teacherflowresid):
                teachloss += torch.mean(refine_out[0]**2) + torch.mean(refine_out[1]**2)

            ## UNIMPORTANT
            if(args.dctnet or args.pcanet):
                if(False):
                    pred_coarse_flow = args.scales[-1] * F.interpolate(pred_flow_pyramid[-1], scale_factor=args.scales[-1]*args.dsstart, mode='bicubic', align_corners=args.align_cornerse)[:,:,:args.patch_size,:args.patch_size]
                    pred_coarse_flow = pred_coarse_flow[:,:,:args.patch_size,:args.patch_size]
            else:
                pred_coarse_flow = 2 ** (args.S_trn) * F.interpolate(pred_flow_pyramid[-1], scale_factor=2 ** (
                    args.S_trn) * args.module_scale_factor, mode='bicubic', align_corners=args.align_cornerse)

            if(False):
                temptemp = args.scales[0] if(args.pcanet)else args.module_scale_factor
                pred_fine_flow = temptemp * F.interpolate(pred_flow_pyramid[0], scale_factor=temptemp*args.dsstart, mode='bicubic',
                                           align_corners=args.align_cornerse)

            
            #print(orthLoss)
            orthLoss = 0.5 * (orthLoss**2)
            #if trainIndex % args.freq_display == 0:
            #   print(rec_loss,smooth_loss,warping_loss,orthLoss)
            if(orthLoss < 0.1):
                total_loss = rec_loss + smooth_loss + warping_loss + args.distillparam * 0.01*flow_distil + args.teachparam* 0.01*teachloss#+ ((orthLoss) if(orthLoss>0.1) else 0)
            else:
                total_loss = rec_loss + smooth_loss + warping_loss + orthLoss + 0.01*flow_distil + args.teachparam*0.01*teachloss


            # compute gradient and do SGD step
            if(once):
                total_loss.backward(retain_graph=True)  # Backpropagate
                #print("Got here")
                once = False
            else:
                total_loss.backward(retain_graph=True)
                #print("GOT HERE")
            optimizer.step()  # Optimizer update


            ##########################################################      
            losses.update(total_loss.item(), 1)
            warp_loss.update(warping_loss.item(),1)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            #################         COMPARISON          ################################
            psnrL = []
            for i in range(args.batch_size):
                if(args.norm_image_warp or args.pcanet):
                    temp = (pred_frameT[i,:].detach().cpu().numpy()+1)/2
                elif(args.dctnet):
                    temp = dct_inverse(pred_frameT[i,:].detach().cpu().numpy(),params=paramsSmallBig[1])
                else:
                    temp = (pred_frameT[i,:].detach().cpu().numpy()+1)/2
                psnrL.append(skimage.metrics.peak_signal_noise_ratio(((frameT[i,:]+1)/2).detach().cpu().numpy(),temp,data_range=1))

            temp_mean = np.mean(psnrL)
            for i in psnrL:
                tbWriter.add_scalar(tag="PSNR",scalar_value=i,global_step=stepper)
                stepper += 1
            tbWriter.add_scalar(tag="Loss",scalar_value=total_loss.item(),global_step=stepper//args.batch_size)
            tbWriter.add_scalar(tag="smoothness loss",scalar_value=smooth_loss.item(),global_step=stepper//args.batch_size)
            tbWriter.add_scalar(tag="warping_loss",scalar_value=warping_loss.item(),global_step=stepper//args.batch_size)

            #print("SHAPES: ",pred_frameT.shape,pred_coarse_flow.shape,pred_fine_flow.shape,frameT.shape,simple_mean.shape,occ_map.shape)

            if trainIndex % args.freq_display == 0:
                progress.print(trainIndex)
                if(args.TOptimization):
                    print("T param: ",model_net.vfinet.T_param.item(), " Learning rate: ",scheduler.get_last_lr())
                if(True):
                    # batch_images = get_batch_images(args, save_img_num=args.save_img_num,
                    #                        save_images=[pred_frameT, pred_coarse_flow, pred_fine_flow, frameT,
                    #                                     simple_mean, occ_map])
                    if(args.ds_normInput):
                        pred_frameT_pyramid = [F.interpolate(pred_frameT_pyramid[i], scale_factor=(2 ** i) ,mode='bicubic', align_corners=args.align_cornerse) for i in range(len(pred_frameT_pyramid))]
                    pyramid_images = get_pyramid_images(args,save_img_num=args.save_img_num,save_images=[i for i in pred_frameT_pyramid]+[i for i in pred_flow_pyramid],frameT=frameT,mean_im=simple_mean) #,refine_out=refine_out
                    
                    ########## COMMENTED THIS LINE OUT #################################################################
                    #temp_path = args.log_dir+"/exp_" + str(args.exp_num)+ "/usual_log"
                    temp_path_pyramid = args.log_dir+"/exp_" + str(args.exp_num)+ "/pyramid"
                    #check_folder(temp_path)
                    check_folder(temp_path_pyramid)
                    #cv2.imwrite(os.path.join(temp_path, '{:03d}_{:04d}_training.png'.format(epoch, trainIndex)), batch_images)
                    cv2.imwrite(os.path.join(temp_path_pyramid, '{:03d}_{:04d}_training.png'.format(epoch, trainIndex)), pyramid_images)

            if(args.directly_save_model):
                best_PSNR = 1
                testLoss = 1
                testPSNR = 1
                combined_state_dict = {
                'net_type': args.net_type,
                'last_epoch': epoch,
                'batch_size': args.batch_size,
                'trainLoss': losses.avg,
                'testLoss': testLoss,
                'testPSNR': testPSNR,
                'best_PSNR': best_PSNR,
                'paramsPCA': model_net.params if(args.pcanet)else None,
                'used_pcas': model_net.used_pcas if(args.pcanet)else None,
                'state_dict_Model': model_net.state_dict(),
                'state_dict_Optimizer': optimizer.state_dict(),
                'state_dict_Scheduler': scheduler.state_dict()}
                SM.save_best_model(combined_state_dict, True)
            # RESUMES DIRECTLY TO VALIDATION!!!
            #break
        # Error of bil and DCT pictures

        print("Time needed for epoch (min): ",(time.time()-start_time_epoch)/60)
        if(args.warping_loss):
            print("Warping loss alpha: ",epochAlpha)
        # Epoch Close UP

        if epoch >= args.lr_dec_start:
            scheduler.step()

        # best_PSNR = 1
        # testLoss = 1
        # testPSNR = 1
        # combined_state_dict = {
        # 'net_type': args.net_type,
        # 'last_epoch': epoch,
        # 'batch_size': args.batch_size,
        # 'trainLoss': losses.avg,
        # 'testLoss': testLoss,
        # 'testPSNR': testPSNR,
        # 'best_PSNR': best_PSNR,
        # 'state_dict_Model': model_net.state_dict(),
        # 'state_dict_Optimizer': optimizer.state_dict(),
        # 'state_dict_Scheduler': scheduler.state_dict()}
        # SM.save_best_model(combined_state_dict, False)  

        if(args.no_validation):
            best_PSNR = 1
            testLoss = 1
            testPSNR = 1
            combined_state_dict = {
            'net_type': args.net_type,
            'last_epoch': epoch,
            'batch_size': args.batch_size,
            'trainLoss': losses.avg,
            'testLoss': testLoss,
            'testPSNR': testPSNR,
            'best_PSNR': best_PSNR,
            'paramsPCA': model_net.params if(args.pcanet)else None,
            'used_pcas': model_net.used_pcas if(args.pcanet)else None,
            'state_dict_Model': model_net.state_dict(),
            'state_dict_Optimizer': optimizer.state_dict(),
            'state_dict_Scheduler': scheduler.state_dict()}
            SM.save_best_model(combined_state_dict, False)    
            continue

        #### EVALUATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if (epoch + 1) % 10 == 0 or epoch==0:
        val_multiple = 4 if args.dataset == 'X4K1000FPS' or args.dataset=="Inter4K" else 2
        print('\nEvaluate on test set exp{} (validation while training) with multiple = {}'.format(args.exp_num,val_multiple))
        postfix = '_val_' + str(val_multiple) + '_S_tst' + str(args.S_tst)
        testLoss, testPSNR, testSSIM, final_pred_save_path,_ = test(valid_loader, model_net, criterion, epoch, args,
                                                                  device, multiple=val_multiple, postfix=postfix,
                                                                  validation=True,val_tracker=valTB_tracker)

        # remember best best_PSNR and best_SSIM and save checkpoint
        print("best_PSNR : {:.3f}, testPSNR : {:.3f}".format(best_PSNR, testPSNR))
        best_PSNR_flag = testPSNR > best_PSNR
        best_PSNR = max(testPSNR, best_PSNR)
        # save checkpoint.
        combined_state_dict = {
            'net_type': args.net_type,
            'last_epoch': epoch,
            'batch_size': args.batch_size,
            'trainLoss': losses.avg,
            'testLoss': testLoss,
            'testPSNR': testPSNR,
            'best_PSNR': best_PSNR,
            'paramsPCA': model_net.params if(args.pcanet)else None,
            'used_pcas': model_net.used_pcas if(args.pcanet)else None,
            'state_dict_Model': model_net.state_dict(),
            'state_dict_Optimizer': optimizer.state_dict(),
            'state_dict_Scheduler': scheduler.state_dict()}

        # Saves the LATEST model!! If best model, it overwrites current best model!
        SM.save_best_model(combined_state_dict, best_PSNR_flag)

        # EPOCH SAVES
        if (epoch + 1) % 10 == 0:
            SM.save_epc_model(combined_state_dict, epoch)
        SM.write_info('{}\t\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.4}\n'.format(epoch, losses.avg,warp_loss.avg,testLoss ,testPSNR, best_PSNR))
    tbWriter.flush()
    tbWriter.close()
    # Until here, everything is done for each epoch
    print("------------------------- Training has been ended. -------------------------\n")
    print("information of model:", args.model_dir)
    print("best_PSNR of model:", best_PSNR)


def test(test_loader, model_net, criterion, epoch, args, device, multiple, postfix, validation,val_tracker=0):
    batch_time = AverageClass('Time:', ':6.3f')
    pred_time = AverageClass('PredTime:', ':6.3f')
    losses = AverageClass('testLoss:', ':.4e')
    PSNRs = AverageClass('testPSNR:', ':.4e')
    PSNRsList = [AverageClass('testPSNR:', ':.4e') for i in range(multiple-1)]
    SSIMs = AverageClass('testSSIM:', ':.4e')
    args.divide = 2 ** (args.S_tst) * args.module_scale_factor * 4
    
    #torch.backends.cudnn.benchmark = True
    #torch.backends.cudnn.enabled = True
    
    sizesDS = {"Inter4K88":[2160,3840],"Inter4K816":[2160,3840]  ,"Xiph": [2160,4096], "Adobe240": [720,1280], "X4K1000FPS": [2160,4096],"Vimeo": [256,448]}
    # progress = ProgressMeter(len(test_loader), batch_time, accm_time, losses, PSNRs, SSIMs, prefix='Test after Epoch[{}]: '.format(epoch))
    progress = ProgressMeter(len(test_loader), PSNRs, SSIMs,pred_time, prefix="exp "+str(args.exp_num) +' Test after Epoch[{}]: '.format(epoch))

    multi_scale_recon_loss = criterion[0]

    torch.backends.cudnn.enabled = True 

    torch.backends.cudnn.benchmark = True

    # switch to evaluate mode
    model_net.eval()
    if(args.halfXVFI):
        model_net = model_net.half()


    skipList = []
    val_loss_q = []
    print("------------------------------------------- Test "+ ("X4K1000FPS" if (validation)else args.dataset) + " ----------------------------------------------")
    print("Multiple: ",multiple)
    with torch.no_grad():
        #start_time = time.time()
        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(test_loader):
            # Shape of 'frames' : [1,C,T+1,H,W]
            if(args.jumptotest>1):
                if(testIndex<args.jumptotest):
                    continue

            if(len(frames.shape)<2):
                skipList.append(testIndex)
                continue
            frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            It_Path, I0_Path, I1_Path = frameRange

            #print(I0_Path,I1_Path)
            #print(It_Path,t_value)
            #a = np.squeeze(frameT.detach().cpu().numpy())
            #a = denorm255_np(np.transpose(a, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]
            #cv2.imwrite('ownTextFiles/temp.png', a)
            

            torch.cuda.empty_cache()
            frameTList = []

            start_time = time.time()
            torch.cuda.empty_cache()
            if (testIndex % (multiple - 1)) == 0:
                torch.cuda.empty_cache()
                B, C, T, H, W = frames[:, :, :-1, :, :].size()
                _,_,_,OH,OW = frames[:, :, :-1, :, :].size()
                if(args.resbefaf>1):
	                input_frames = F.interpolate(frames[:, :, :-1, :, :].permute(0,2,1,3,4).reshape(B*T,C,H,W), 
	                	scale_factor=1/args.resbefaf, mode='bicubic',align_corners=args.align_cornerse).reshape(B,T,C,H//args.resbefaf,W//args.resbefaf).permute(0,2,1,3,4)		
                else:
                	input_frames = frames[:, :, :-1, :, :]
                # if(args.pcanet):
                #     input_gpu = (input_frames.clone().to(device))

                B, C, T, H, W = input_frames.size()
                #input_frames = Variable(input_frames.to(device))
                # This padding is not needed for DCTNET, as I do it already (use for XVFI is simply, that the image is finely divided through the 1/(2^s) downscaling operations)
                if(args.dctnet or args.pcanet):
                    # Input frames Padding  [1, 3, 2, 2160, 4096]
                    temp_shap = input_frames.shape
                    input_frames = input_frames.reshape(temp_shap[0],-1,temp_shap[3],temp_shap[4])
                    div_pad = (2**args.S_tst)*8 if(args.phase=="test") else (2**args.S_trn)*8
                    div_pad *= args.dsstart
                    H_padding = (div_pad - H % div_pad) % div_pad
                    W_padding = (div_pad - W % div_pad) % div_pad
                    #H_padding = 0 if(validation)else (144 if(args.S_tst == 5) else 16)   
                    #W_padding = 0 if(validation)else 0 
                    paddingmode = args.padding #"constant" if(args.pcanet)else "reflect"
                    input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), paddingmode)
                    if(args.dataset == "Adobe240"):
                        W_padding = 0
                        H_padding = 48
                        assert False
                        input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), paddingmode)
                    input_frames = input_frames.reshape(temp_shap[0],temp_shap[1],temp_shap[2],temp_shap[3]+H_padding,temp_shap[4]+W_padding)    
                else:
                    H_padding = (args.divide - H % args.divide) % args.divide
                    W_padding = (args.divide - W % args.divide) % args.divide
                    if H_padding != 0 or W_padding != 0:
                        input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), "constant")
                
                #print("BEfore preprocessing")
                #time.sleep(3)
                # POTENZIELLES PROBLEM TODO
                # hier koennte ich input frames auch nur alle 8 iterationen preprocessen!!!! MAch ich hier mehrmals DCT auf inputframes?????????!?!?!?????????!?!!!!!
                if(args.pcanet):
                    input_gpuList = preprocessing(args,input_frames,frameTList,device,frameT,train=False,onlyPCA=True,model_net=model_net)
                    #input_gpu = [(input_frames.to(device))] 
                    B,C,T,H,W = input_frames.shape
                    if(args.smoothimages):
                        input_gpu = [model_net.vfinet.get_gaussian_kernel(sigma=(( args.scales[i]/args.scales[0])-1)/2)(F.interpolate(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W), scale_factor=args.scales[0]/ (args.scales[i]),mode='bicubic', 
                            align_corners=args.align_cornerse)).to(device).reshape(B,T,C,int(H*(args.scales[0]/ (args.scales[i]))),int(W*(args.scales[0]/ ( args.scales[i])))).permute(0,2,1,3,4) if(i!=0 ) else input_frames.to(device)  for i in range(args.S_tst+1)]
                    else:
                        input_gpu = [F.interpolate(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W), scale_factor=args.scales[0]/ (args.scales[i]),mode='bicubic', 
                        align_corners=args.align_cornerse).to(device).reshape(B,T,C,int(H*(args.scales[0]/ (args.scales[i]))),int(W*(args.scales[0]/ ( args.scales[i])))).permute(0,2,1,3,4) if(i!=0 ) else input_frames.to(device)  for i in range(args.S_tst+1)]
                    

                    frameTList = []
                    #frameT = frameT.to(device)
                elif(args.dctnet and (not args.norm_image_warp)):
                    torch_prints(input_frames)
                    input_gpu_small,input_gpu_big,pcaSmallBig,paramsSmallBig = preprocessing(args,input_frames,frameTList,device,frameT,train=False)
                elif(args.norm_image_warp):
                    input_gpu_small,input_gpu_big,pcaSmallBig,paramsSmallBig = preprocessing(args,input_frames,frameTList,device,frameT,train=False)
                    input_gpu = input_frames.to(device) # as to_dctpca already brings it on GPU
                else:
                    input_gpu = input_frames.to(device) # as to_dctpca already brings it on GPU
                
            
            t_value = Variable(t_value.to(device))
            torch.cuda.empty_cache()
            pred_starttime = time.time()
            #print("BEfore modelnet t-value: ",t_value.item())

            #print("before model_net forward")
            #time.sleep(3)
            #temstarttime= time.time()
            if(args.pcanet): #predframeT already the right size
                pred_frameT,fine_flow = model_net(input_gpuList, t_value,normInput=[im.clone() for im in input_gpu],is_training=False,validation=validation)
            elif(args.dctnet and (not args.norm_image_warp)):
                pred_frameT,fine_flow = model_net(input_gpu_small,input_gpu_big, t_value,pcaSmallBig,paramsSmallBig,is_training=False)
            elif(args.norm_image_warp):
                pred_frameT,fine_flow = model_net(input_gpu_small,input_gpu_big, t_value,pcaSmallBig,paramsSmallBig,is_training=False,normInput=input_gpu)
            else:
                if(args.halfXVFI):
                    input_gpu = input_gpu.half()
                pred_frameT = model_net(input_gpu, t_value, is_training=False)
            
            # stats = torch.cuda.memory_stats()
            # peak_bytes_requirement = stats["allocated_bytes.all.peak"]
            # print(f"Peak memory requirement: {peak_bytes_requirement / 1024 ** 3:.2f} GB")
            #print("just model call: ",time.time()-temstarttime)

            if(args.resbefaf>1):
                pred_frameT = F.interpolate(pred_frameT, scale_factor=args.resbefaf, mode='bicubic',align_corners=args.align_cornerse)

            pred_time.update(time.time()-pred_starttime)
            batch_time.update(time.time() - start_time)
            #print("predicted FrameT shape: ",pred_frameT.shape)#[1,3,2160,4096]
            #print("AFTER MODELNET")

            # Calculate loss on PredFrameT and save it
            if(validation):
                rec_loss = args.rec_lambda * multi_scale_recon_loss(pred_frameT.detach().cpu().clone(),frameT.detach().cpu().clone())
                tbWriter = val_tracker["tbWriter"]
                tbWriter.add_scalar(tag="Rec Validation Loss",scalar_value=rec_loss.item(),global_step=val_tracker["stepper"])
                val_loss_q.append(rec_loss.item())
                val_tracker["stepper"] += 1

            # FRAME HANDLING CUTTING AND TRANSFORMING
            if(args.pcanet or args.norm_image_warp):
                if((args.dataset=="Xiph" or args.dataset=="X4K1000FPS" or args.dataset=="Inter4K88" or args.dataset=="Inter4K816") and not validation and not args.xiph2k and not args.xtest2k):
                    tempsize = sizesDS[args.dataset]
                    assert OH==tempsize[0] and OW==tempsize[1]

                
                pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())[:,:OH,:OW]
                frameT = np.squeeze(frameT.detach().cpu().numpy())
                #print(pred_frameT.shape,frameT.shape)
            elif(not args.dctnet):
                if H_padding != 0 or W_padding != 0:
                    pred_frameT = pred_frameT[:, :, :H, :W]
                #print("pred frame shape : ",pred_frameT.shape)
                pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())
                frameT = np.squeeze(frameT.detach().cpu().numpy())
            else:
                pred_frameT = (dct_inverse(pred_frameT[0,:,:,:].detach().cpu().numpy(),params=paramsSmallBig[1])*2)-1 # make it [-1,1] as [0,1] comes in
                frameT = frameT[0,:,:,:].detach().cpu().numpy() #dct_inverse(frameT[0,:,:,:].detach().cpu().numpy(),params=paramsSmallBig[1])



            if args.phase != 'test_custom':
                #test_loss = args.rec_lambda * multi_scale_recon_loss(pred_frameT, frameT)

                
                #numpy_prints(pred_frameT,"pred_frameT")
                #numpy_prints(frameT,"frameT")
                """ compute PSNR & SSIM """ # From output And Ground Truth!
                output_img = np.around(denorm255_np(np.transpose(pred_frameT, [1, 2, 0])))  # [h,w,c] and [-1,1] to [0,255]
                target_img = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]

                #print("output/target",output_img.shape,target_img.shape)
                outputest = True
                if(outputest ): #and (args.dataset=="Xiph" or args.dataset=="X4K1000FPS" or args.dataset=="Inter4K")
                    testpredspath = os.path.join("log_dir",args.model_dir,args.dataset,str(testIndex//(multiple-1))) #str(int(t_value*8))
                    check_folder(testpredspath)
                    cv2.imwrite(os.path.join(testpredspath,str(int(t_value*multiple))+".png"),output_img)
                    cv2.imwrite(os.path.join(testpredspath,"GT"+str(int(t_value*multiple))+".png"),target_img)
                    #cv2.imwrite("tempTest/target.png",target_img)
                    #print(input_frames[0,0,:,:,:].permute(1,2,0).detach().cpu().numpy().shape)
                    if(testIndex % (multiple-1) == 0):
                        tempsize = [OH,OW]#sizesDS[args.dataset]
                        cv2.imwrite(os.path.join(testpredspath,"0.png"),((input_frames[0,:,0,:tempsize[0],:tempsize[1]].permute(1,2,0).detach().cpu().numpy()+1)/2) * 255)
                        cv2.imwrite(os.path.join(testpredspath,str(multiple)+".png"),((input_frames[0,:,1,:tempsize[0],:tempsize[1]].permute(1,2,0).detach().cpu().numpy()+1)/2) * 255)
                        
                #print("TEST PRINTS FOR FRAME T")
                #numpy_prints(output_img,"output img")
                #numpy_prints(target_img,"target img")
                test_psnr = psnr(target_img, output_img,args)
                test_ssim = ssim_bgr(target_img, output_img)  ############### CAUTION: calculation for BGR

                """ save frame0 & frame1 """
                if validation:
                    epoch_save_path = os.path.join(args.test_img_dir, args.model_dir, 'latest' + postfix)
                else:
                    epoch_save_path = os.path.join(args.test_img_dir, args.model_dir,
                                                   'epoch_' + str(epoch).zfill(5) + postfix)
                check_folder(epoch_save_path)
                scene_save_path = os.path.join(epoch_save_path, scene_name[0])
                check_folder(scene_save_path)

                #a = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]
                #cv2.imwrite('ownTextFiles/1.png', a)

                #print("scenepath: ",scene_save_path)
                save_it = False
                if(save_it):
                    if (testIndex % (multiple - 1)) == 0:
                        save_input_frames = frames[:, :, :-1, :, :]
                        cv2.imwrite(os.path.join(scene_save_path, I0_Path[0]),
                                    np.transpose(np.squeeze(denorm255_np(save_input_frames[:, :, 0, :, :].detach().numpy())),
                                                 [1, 2, 0]).astype(np.uint8))
                        cv2.imwrite(os.path.join(scene_save_path, I1_Path[0]),
                                    np.transpose(np.squeeze(denorm255_np(save_input_frames[:, :, 1, :, :].detach().numpy())),
                                                 [1, 2, 0]).astype(np.uint8))

                    cv2.imwrite(os.path.join(scene_save_path, It_Path[0]), output_img.astype(np.uint8))

                    #######################     Save FLOW       #############################
                    # Upscale
                #a = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]
                #cv2.imwrite('ownTextFiles/2.png', a)
                if(not validation and not args.dataset=='Vimeo' and False):
                    if(not args.pcanet):
                        fine_flow = torch.zeros((1,3,2160,4096))
                    else:
                        fine_flow = fine_flow.detach().cpu()
                        torch.cuda.empty_cache()
                        
                        flowMul = args.scales[0]
                        fine_flow = flowMul * F.interpolate(fine_flow, scale_factor=flowMul, mode='bicubic',
                                               align_corners=args.align_cornerse)
                        fine_flow = fine_flow[:,:,:sizesDS[args.dataset][0],:sizesDS[args.dataset][1]]  
                    # Flow to image and difference pic
                    #a = np.squeeze(frameT.detach().cpu().numpy())
                    
                    #a = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]
                    #cv2.imwrite('ownTextFiles/3.png', a)
                    pic_flow,diff_pic = get_test_pred_flow(args,fine_flow,output_img,target_img)
                    # Path
                    temp_path = args.log_dir+"/exp_" + str(args.exp_num)+"/Test_flow_preds"
                    temp_path = os.path.join(temp_path, scene_name[0])
                    check_folder(temp_path)
                    cv2.imwrite(os.path.join(temp_path, f"target{t_value.item():.4f}.png"), target_img)
                    cv2.imwrite(os.path.join(temp_path,f"output{t_value.item():.4f}.png"), output_img)
                    cv2.imwrite(os.path.join(temp_path,f"flow{t_value.item():.4f}.png"), pic_flow)
                    cv2.imwrite(os.path.join(temp_path,f"diff{t_value.item():.4f}.png"), diff_pic)
                    
                    first = denorm255_np(np.transpose(np.squeeze(input_frames[:,:,0,:2160,:4096].detach().cpu().numpy()), [1, 2, 0]))
                    cv2.imwrite(os.path.join(temp_path,f"first{0:.4f}.png"), first)
                    second = denorm255_np(np.transpose(np.squeeze(input_frames[:,:,1,:2160,:4096].detach().cpu().numpy()), [1, 2, 0]))
                    cv2.imwrite(os.path.join(temp_path,f"second{1:.4f}.png"), second)

                    #a = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]
                    #cv2.imwrite('ownTextFiles/4.png', a)


                # measure
                if(validation):
                    losses.update(rec_loss.item(), 1)
                if(not validation and (args.dataset == "Inter4K88" or args.dataset == "Inter4K816")):
                	PSNRsList[int(t_value*multiple)-1].update(test_psnr,1)
                PSNRs.update(test_psnr, 1)
                SSIMs.update(test_ssim, 1)

                # measure elapsed time
                #batch_time.update(time.time() - start_time)
                #start_time = time.time()

                if (testIndex % (multiple - 1)) == multiple - 2:
                    progress.print(testIndex)
                    if(not validation and  (args.dataset == "Inter4K88" or args.dataset == "Inter4K816")):
	                    printstring = " ".join([str(index)+": "+str(ttime.avg)+ " || " for index,ttime in enumerate(PSNRsList)])
	                    print(printstring)
                    if(args.stoptestat != -1):
                        if(testIndex > args.stoptestat):
                            break
            # else:
            #     epoch_save_path = args.custom_path
            #     scene_save_path = os.path.join(epoch_save_path, scene_name[0])
            #     pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())
            #     output_img = np.around(denorm255_np(np.transpose(pred_frameT, [1, 2, 0])))  # [h,w,c] and [-1,1] to [0,255]
            #     print(os.path.join(scene_save_path, It_Path[0]))
            #     cv2.imwrite(os.path.join(scene_save_path, It_Path[0]), output_img.astype(np.uint8))

            #     losses.update(0.0, 1)
            #     PSNRs.update(0.0, 1)
            #     SSIMs.update(0.0, 1)

        if(validation):
            print("Validation loss mean: ", torch.mean(torch.as_tensor(val_loss_q)))
        print("-----------------------------------------------------------------------------------------------")
    print("These indices were skipped: ",skipList)
    return losses.avg, PSNRs.avg, SSIMs.avg, epoch_save_path,PSNRsList


if __name__ == '__main__':
    main()
