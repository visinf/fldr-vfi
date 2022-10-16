# This code contains parts of XVFInet from Sim et al. (https://github.com/JihyongOh/XVFI) 
# Their extensive code and Dataset were crucial for this.

import argparse, os, shutil, time, random, torch, cv2, datetime, torch.utils.data, math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
import cupy as cp
import scipy.fft as scF
import skimage.metrics

from pca_comp import DCTParams,to_pca 
from useful import ScaleIt,torch_prints,numpy_prints,MyPWC,distillation_loss,getmodelconfig

from torch.autograd import Variable
from torchvision import utils
from utils import *
from fLDRnet import *
from collections import Counter
import sys
from skimage.transform import rescale

from torch.utils.tensorboard import SummaryWriter

def parse_args():
    desc = "PyTorch implementation for XVFI"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--gpu', type=int, default=0, help='gpu index')
    parser.add_argument('--net_type', type=str, default='fLDRnet', choices=['fLDRnet'], help='The type of Net')
    parser.add_argument('--exp_num', type=int, default=1, help='The experiment number')
    parser.add_argument('--phase', type=str, default='test', choices=['train', 'test', 'test_custom', 'metrics_evaluation',])
    parser.add_argument('--continue_training', action='store_true', default=False, help='continue the training')

    



    parser.add_argument('--text_dir', type=str, default='./text_dir', help='text_dir path')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint_dir', help='checkpoint_dir')
    parser.add_argument('--log_dir', type=str, default='./log_dir', help='Directory name to save training logs')

    parser.add_argument('--dataset', default='X4K1000FPS', choices=['X4K1000FPS', 'Vimeo',"Inter4K",'Xiph'],
                        help='Training/test Dataset')

    


    superprefix = './../../' 

    prefix = superprefix +  'X-Train/'

    parser.add_argument('--x_train_data_path', type=str, default=prefix+'train')
    parser.add_argument('--x_val_data_path', type=str, default=prefix+'val')
    parser.add_argument('--x_test_data_path', type=str, default=prefix+'test')


    parser.add_argument('--vimeo_data_path', type=str, default=superprefix+'vimeo_triplet') 
    parser.add_argument('--xiph_data_path', type=str, default="../XVFI-main/XiphDataset/netflix") 
    parser.add_argument('--inter4k_data_path', type=str, default="../XVFIM-main/inter4K/Inter4KNewTestset/") 
    M

    ################################        My Hyperparameters      #############################################
    


    parser.add_argument('--validation_patch_size', type=int, default=512, help='patch size in validation')
    parser.add_argument('--test_patch_size', type=int, default=-1, help='patch size in test. If -1 no patching is done') 
    parser.add_argument('--pin_memory_train', action="store_true", help='faster?')
    parser.add_argument('--pin_memory_test', action="store_true", help='faster?')
    parser.add_argument('--test5scales', action="store_true", help='faster?')
    parser.add_argument('--test6scales', action="store_true", help='faster?')
    parser.add_argument('--test7scales', action="store_true", help='faster?')
    parser.add_argument('--test4scales', action="store_true", help='faster?')
    parser.add_argument('--test3scales', action="store_true", help='faster?')


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
    parser.add_argument('--pcanet', action="store_true", help='Just PCA conversion!')
    parser.add_argument('--net_object', default=DCTXVFInet, choices=[DCTXVFInet], help='The type of Net')
    

    # GOOD
    parser.add_argument('--ds_normInput', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--dctvfi_nf',  type=int, default=16, help='number of feature maps put into Net per imagechannel')
    parser.add_argument('--scales' ,default=[4,8,16,32,64,128], nargs='+', help='<Required> Set flag')
    parser.add_argument('--fractions' ,default=[1,4,16,64,256,1024], nargs='+', help='<Required> Set flag')
    
    parser.add_argument('--ref_feat_extrac', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--maskLess', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--imageUpInp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--allImUp',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--ExacOneEV',   action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--outMaskLess',   action="store_true", help='no features inputted at refinement step')
    
    parser.add_argument('--papermodel',   action="store_true", help='no features inputted at refinement step')
        
    
    
    parser.add_argument('--meanVecParam', action="store_false", help='no features inputted at refinement step')

    # other
    parser.add_argument('--align_cornerse',action="store_true",help='no features inputted at refinement step')
    parser.add_argument('--takeBestModel',action="store_false",help='no features inputted at refinement step')
    parser.add_argument('--testmessage', type=str, default="",help='no features inputted at refinement step')
    

    # Losses
    parser.add_argument('--warping_loss', action="store_true", help='no features inputted at refinement step')
    parser.add_argument('--warp_alpha',  type=float, default=0.5, help='The initial learning rate')
    parser.add_argument('--endflowwarploss',  action="store_true", help='The initial learning rate')
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
    
    

    # Get what you want
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


    

    return check_args(parser.parse_args())


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --text_dir
    check_folder(args.text_dir)

    # --log_dir
    check_folder(args.log_dir)

 

    return args


def main():
    args = parse_args()

    if(args.papermodel):
        getmodelconfig(args)
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
    args.padding = "reflect" if(args.pcanet )else "constant"

    if(args.phase == "Train"):
        args.tempAdamfix = True
    if(len(args.scales) != len(args.fractions)):
        raise Exception("Scales and Fractions array don't have the same length!")
    if(args.flowtest):
        args.continue_training = True

    if(args.maxmin_vec and args.mean_vector_norm):
        sys.exit()
    
    if args.dataset != 'X4K1000FPS':
        args.multiple = 2
    
    

    assert not args.ExacOneEV or args.allImUp 
    assert  (args.imageUpInp == (not args.ExacOneEV )) or (not args.imageUpInp) and not args.ExacOneEV

    if(args.pcanet):
        assert args.S_trn == args.S_tst or args.moreTstSc
        args.takeBestModel = True
    else:
        args.takeBestModel = False
    


    print("Exp:", args.exp_num)
    args.model_dir = args.net_type + '_' + args.dataset + '_exp' + str(
        args.exp_num) 

    if args is None:
        exit()
    for arg in vars(args):
        print('# {} : {}'.format(arg, getattr(args, arg)))
    device = torch.device(
        'cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu') 
    torch.cuda.set_device(device)  

    print('Available devices: ', torch.cuda.device_count())
    print('Current cuda device: ', torch.cuda.current_device())
    print('Current cuda device name: ', torch.cuda.get_device_name(device))
    if args.gpu is not None:
        print("Use GPU: {} is used".format(args.gpu))

    

    """ Initialize a model """
    model_net = args.net_object(args).apply(weights_init).to(device) 
    criterion = [set_rec_loss(args).to(device),set_smoothness_loss_forward(args=args).to(device) if(args.ownsmooth)else set_smoothness_loss().to(device),set_warping_loss(args).to(device),set_warping_loss_endflow(args).to(device)]

    
    # Parameter Print
    print("Total Parameters:           ",sum(p.numel() for p in model_net.parameters()))
    print("Total learnable Parameters: ",sum(p.numel() for p in model_net.parameters() if p.requires_grad))
    args.parameters = sum(p.numel() for p in model_net.parameters() if p.requires_grad)
    SM = save_manager(args)
    
    cudnn.benchmark = True

    
    if args.phase == "train":
        train(model_net, criterion, device, SM, args)
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

    
    postfix = '_final_x' + str(args.multiple) + '_S_tst' + str(args.S_tst)

    if args.phase != "metrics_evaluation":
        print("\n-------------------------------------- Final Test starts -------------------------------------- ")
        print('Evaluate on test set (final test) with multiple = %d ' % (args.multiple))

        for i in args.testsets:
            args.dataset = i
            temMultiple = {"X4K1000FPS": 8,"XTest2KC":8,"Inter4K88":8,"Inter4K816":8,"Xiph": 2,"Xiph2KC":2,"Vimeo":2 ,"Adobe240": 8,"HD":4}
            final_test_loader = get_test_data(args, args.dataset,multiple=temMultiple[i],
                                              validation=False,specific=i)  


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

   


    print("------------------------- Test has been ended. -------------------------\n")
    print("Exp: ", args.exp_num)


def preprocessing(args,input_frames,frameTList,device,frameT,train=False,onlyPCA=False,model_net=None):

    shap = input_frames.shape
    input_frames = input_frames.reshape(shap[0],-1,shap[3],shap[4])
    

    shap = input_frames.shape
    H,W = shap[2:4]
    data_used = 0.5 if(args.phase=="train")else 0.01

    params = model_net.params
    diff_scales = len(args.scales) - len(params)
    for i in range(diff_scales ):
    	params.append(DCTParams(wiS=8,components_fraction=1/4,data_used=data_used ) )
    input_gpuList = []
    for l in range(len(params)):
    	input_gpuList.append( torch.zeros((shap[0],int(args.img_ch*2*(params[l].wiS**2)*params[l].components_fraction) ,H//params[l].wiS,W//params[l].wiS),device=device))
                  

    mempool = cp.get_default_memory_pool()
    mempool.free_all_blocks()


    return input_gpuList


def once_preprocessed(input_frames,model_net,args,device):
    H,W = input_frames.shape[2:4]
    data_used = 0.5 if(args.phase=="train")else 0.01
    params = []
    for i in range(len(args.scales)):    
        temFrac = args.fractions[args.scales.index(8)]
        params.append(DCTParams(wiS=8,components_fraction=1/temFrac,data_used=data_used ) )
        

    print(params)
    model_net.save_params(params)
    
    all_pcas = []


    if(args.simpleEVs):
        _,all_pcas = to_pca(input_frames.permute(1,0,2,3).reshape(-1,H,W),params[0],components_fraction=0,args=args)
    else:
        for index,i in enumerate(params):
            if(args.allImUp and args.scales[index] != 8):
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
  

def train(model_net, criterion, device, save_manager, args):

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

    
    optimIn = [{"params": model_net.ev_params, "lr": args.init_lr*args.evlr}
                ,{"params": model_net.base_modules.parameters()}]
    if(args.noEVOptimization):
        optimIn = [{"params": model_net.base_modules.parameters()}]
    optimizer = optim.Adam(optimIn, lr=args.init_lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)  # optimizer

    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_dec_fac)

    last_epoch = 0
    best_PSNR = 0.0

    
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
            
            model_net.pick_norm_vec(checkpoint['used_pcas'])
        model_net.save_params(checkpoint['paramsPCA'])
        print("Optimizer and Scheduler have been reloaded. ")
        

    scheduler.milestones = Counter(args.lr_milestones)
    scheduler.gamma = args.lr_dec_fac
    print("scheduler.milestones : {}, scheduler.gamma : {}".format(scheduler.milestones, scheduler.gamma))
    if(args.flowtest):
        start_epoch = 0
    else:
        start_epoch = last_epoch



    model_net.train()
    model_net.to(device)

    start_time = time.time()
    if(not args.continue_training):
        SM.write_info('Epoch\ttrainLoss\tWarpLoss\tValLoss\ttestPSNR\tbest_PSNR\n')
    print("[*] Training starts")


    valid_loader = get_test_data(args,"X4K1000FPS" if(args.dataset=="Inter4K88" or args.dataset=="Inter4K816")else args.dataset, multiple=4, validation=True)  # multiple is only used for X4K1000FPS
    
    tempIndex = 0
    if(args.TOptimization):
        for param in model_net.ev_params:
            param.requires_grad = False
        for index,param in enumerate(model_net.base_modules.named_parameters()):
            if(param[0] != "1.T_param"):
                param[1].requires_grad = False
            else:
                param[1].requires_grad = True
            

     

    once = True
    for epoch in range(start_epoch, args.epochs):
        train_loader,_ = get_train_data(args,
                                      max_t_step_size=32,device=device) 

        batch_time = AverageClass('batch_time[s]:', ':6.3f')
        losses = AverageClass('Loss:', ':.4e')
        warp_loss = AverageClass('Warp:',':.4e')
        progress = ProgressMeter(len(train_loader), batch_time, losses, prefix="Epoch: [{}]".format(epoch))

        print('Start epoch {} at [{:s}], learning rate : [{}]'.format(epoch, (str(datetime.now())[:-7]),
                                                                      optimizer.param_groups[0]['lr']))
        print("Learning rates: ", [i["lr"] for i  in optimizer.param_groups])

        start_time_epoch = time.time()
        
        for trainIndex, (frames, t_value) in enumerate(train_loader):



            
            
            input_frames = frames[:, :, :-1, :] 
            B, C, T, H, W = input_frames.shape
            frameT = frames[:, :, -1, :]  
            frameTList = []

            if(once  and not args.continue_training ):
                once_preprocessed(input_frames[0,:,:,:,:].detach(),model_net,args,device)
            
            input_gpuList = preprocessing(args,input_frames,frameTList,device,frameT,train=True,onlyPCA=True,model_net=model_net)
            
            
            input_gpu = [F.interpolate(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W), scale_factor=args.scales[0]/ args.scales[i],mode='bicubic', 
                    align_corners=args.align_cornerse).to(device).reshape(B,T,C,int(H*(args.scales[0]/ args.scales[i])),int(W*(args.scales[0]/ args.scales[i]))).permute(0,2,1,3,4) if(i!=0) else input_frames.to(device)  for i in range(args.S_trn+1)]#print(torch.mean(input_frames[1,:,0,:,:].clone()))
            
            frameT = frameT.to(device) 
            frameTList = []
            for i in range(args.S_trn+1):
                frameTList.append(frameT)

            
                
            t_value = t_value.to(device)  
            optimizer.zero_grad()

            pred_frameT_pyramid, pred_flow_pyramid,unref_flow_pyramid, occ_map, simple_mean, endflow = model_net(input_gpuList, t_value,normInput=input_gpu,epoch=epoch,frameT=frameT)
            
            
            rec_loss = 0.0
            smooth_loss = 0.0
            flow_distil = torch.tensor(0.0,device=device)
            warping_loss = torch.tensor(0.0,device=device)
            orthLoss = torch.tensor(0.0,device=device)
            

            for l, pred_frameT_l in enumerate(pred_frameT_pyramid):
                temp = 1/ (2**l)
                temp = args.scales[0]/args.scales[l]
                if(args.TOptimization):
                    tempLoss = torch.mean((pred_frameT_l - F.interpolate(frameTList[l], scale_factor=temp, 
                                                                                       mode='bicubic', align_corners=args.align_cornerse))**2)
                else:
                    tempLoss = multi_scale_recon_loss(pred_frameT_l,F.interpolate(frameTList[l], scale_factor=temp, 
                                                                                   mode='bicubic', align_corners=args.align_cornerse))
                rec_loss += args.rec_lambda * tempLoss
                
            
            if(args.ownsmooth):
                smooth_loss += args.smoothness * smoothness_loss(pred_flow_pyramid[0],
                                                 F.interpolate(input_frames[:,:,0,:].to(device), scale_factor=1 / args.scales[0] ,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse),F.interpolate(input_frames[:,:,1,:].to(device), scale_factor=1 / args.scales[0] ,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse))
            else:
            	smooth_loss += args.smoothness * smoothness_loss(pred_flow_pyramid[0],
                                                 F.interpolate(frameTList[0], scale_factor=1 / args.scales[0] ,
                                                               mode='bicubic',
                                                               align_corners=args.align_cornerse))  # Apply 1st order edge-aware smoothness loss to the fineset level
            
            if(args.forwendflowloss):
                endflowforloss = set_warping_loss_endflow_forward()
                warping_loss += args.warp_alpha * endflowforloss(endflow[0],input_frames[:,:,0,:].to(device),input_frames[:,:,1,:].to(device),t_value)
            if(args.warping_loss):
                fine_unrefined_flow = args.scales[0] * F.interpolate(unref_flow_pyramid[0], scale_factor=args.scales[0], mode='bicubic',align_corners=args.align_cornerse)
                epochAlpha = args.warp_alpha * (1 - torch.exp(-torch.tensor((args.epochs/4-epoch),device=device))) if(epoch <= args.epochs/4)else 0
                warping_loss += epochAlpha * warping_rec_loss(input_frames.to(device),fine_unrefined_flow)
                
            
            input_gpu = 0
            input_gpuList = 0
            in_temp = 0
            torch.cuda.empty_cache()
            
            if(args.orthLoss):
                for index,evs in enumerate(model_net.EVs):
                    if(index == len(args.scales)):
                        continue
                    for kev in range(args.dctvfi_nf):
                        for lev in range(args.dctvfi_nf):
            
                            if(kev == lev or evs == None):
                                continue
                            orthLoss += torch.sum(evs[kev,:] * evs[lev,:])
                assert len(orthLoss.shape) == 0
                assert not orthLoss.isnan(), "orthLoss is nan"
            

            rec_loss /= len(pred_frameT_pyramid)
            pred_frameT = pred_frameT_pyramid[0] 

            

            

            
            orthLoss = 0.5 * (orthLoss**2)
            
            if(orthLoss < 0.1):
                total_loss = rec_loss + smooth_loss + warping_loss  
            else:
                total_loss = rec_loss + smooth_loss + warping_loss + orthLoss 


            # compute gradient and do SGD step
            if(once):
                total_loss.backward(retain_graph=True)  # Backpropagate
                once = False
            else:
                total_loss.backward(retain_graph=True)
            optimizer.step()  # Optimizer update


            
            losses.update(total_loss.item(), 1)
            warp_loss.update(warping_loss.item(),1)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            #################         COMPARISON          ################################
            psnrL = []
            for i in range(args.batch_size):
                temp = (pred_frameT[i,:].detach().cpu().numpy()+1)/2
                psnrL.append(skimage.metrics.peak_signal_noise_ratio(((frameT[i,:]+1)/2).detach().cpu().numpy(),temp,data_range=1))

            

            

            if trainIndex % args.freq_display == 0:
                progress.print(trainIndex)
                if(args.TOptimization):
                    print("T param: ",model_net.vfinet.T_param.item(), " Learning rate: ",scheduler.get_last_lr())
                if(args.save_images):
                    if(args.ds_normInput):
                        pred_frameT_pyramid = [F.interpolate(pred_frameT_pyramid[i], scale_factor=(2 ** i) ,mode='bicubic', align_corners=args.align_cornerse) for i in range(len(pred_frameT_pyramid))]
                    pyramid_images = get_pyramid_images(args,save_img_num=args.save_img_num,save_images=[i for i in pred_frameT_pyramid]+[i for i in pred_flow_pyramid],frameT=frameT,mean_im=simple_mean) 
                    
                    temp_path_pyramid = args.log_dir+"/exp_" + str(args.exp_num)+ "/pyramid"
                    check_folder(temp_path_pyramid)
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
 

        print("Time needed for epoch (min): ",(time.time()-start_time_epoch)/60)
        if(args.warping_loss):
            print("Warping loss alpha: ",epochAlpha)
        # Epoch Close UP

        if epoch >= args.lr_dec_start:
            scheduler.step()

 

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


        val_multiple = 4 if args.dataset == 'X4K1000FPS' or args.dataset=="Inter4K" else 2
        print('\nEvaluate on test set exp{} (validation while training) with multiple = {}'.format(args.exp_num,val_multiple))
        postfix = '_val_' + str(val_multiple) + '_S_tst' + str(args.S_tst)
        testLoss, testPSNR, testSSIM, final_pred_save_path,_ = test(valid_loader, model_net, criterion, epoch, args,
                                                                  device, multiple=val_multiple, postfix=postfix,
                                                                  validation=True,val_tracker=valTB_tracker)


        print("best_PSNR : {:.3f}, testPSNR : {:.3f}".format(best_PSNR, testPSNR))
        best_PSNR_flag = testPSNR > best_PSNR
        best_PSNR = max(testPSNR, best_PSNR)

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


        SM.save_best_model(combined_state_dict, best_PSNR_flag)


        if (epoch + 1) % 10 == 0:
            SM.save_epc_model(combined_state_dict, epoch)
        SM.write_info('{}\t\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.4}\t\t{:.4}\n'.format(epoch, losses.avg,warp_loss.avg,testLoss ,testPSNR, best_PSNR))
    tbWriter.flush()
    tbWriter.close()

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
    


    
    sizesDS = {"Inter4K88":[2160,3840],"Inter4K816":[2160,3840]  ,"Xiph": [2160,4096], "Adobe240": [720,1280], "X4K1000FPS": [2160,4096],"Vimeo": [256,448]}

    progress = ProgressMeter(len(test_loader), PSNRs, SSIMs,pred_time, prefix="exp "+str(args.exp_num) +' Test after Epoch[{}]: '.format(epoch))

    multi_scale_recon_loss = criterion[0]

    torch.backends.cudnn.enabled = True 

    torch.backends.cudnn.benchmark = False

    # switch to evaluate mode
    model_net.eval()


    skipList = []
    val_loss_q = []
    print("------------------------------------------- Test "+ ("X4K1000FPS" if (validation)else args.dataset) + " ----------------------------------------------")
    print("Multiple: ",multiple)
    with torch.no_grad():

        for testIndex, (frames, t_value, scene_name, frameRange) in enumerate(test_loader):

            if(args.jumptotest>1):
                if(testIndex<args.jumptotest):
                    continue

            if(len(frames.shape)<2):
                skipList.append(testIndex)
                continue
            frameT = frames[:, :, -1, :, :]  # [1,C,H,W]
            It_Path, I0_Path, I1_Path = frameRange


            torch.cuda.empty_cache()
            frameTList = []

            start_time = time.time()
            torch.cuda.empty_cache()
            if (testIndex % (multiple - 1)) == 0:
                torch.cuda.empty_cache()
                B, C, T, H, W = frames[:, :, :-1, :, :].size()
                _,_,_,OH,OW = frames[:, :, :-1, :, :].size()
                input_frames = frames[:, :, :-1, :, :]
                B, C, T, H, W = input_frames.size()

                temp_shap = input_frames.shape
                input_frames = input_frames.reshape(temp_shap[0],-1,temp_shap[3],temp_shap[4])
                div_pad = (2**args.S_tst)*8 if(args.phase=="test") else (2**args.S_trn)*8
                
                H_padding = (div_pad - H % div_pad) % div_pad
                W_padding = (div_pad - W % div_pad) % div_pad

                paddingmode = args.padding
                input_frames = F.pad(input_frames, (0, W_padding, 0, H_padding), paddingmode)
                input_frames = input_frames.reshape(temp_shap[0],temp_shap[1],temp_shap[2],temp_shap[3]+H_padding,temp_shap[4]+W_padding)    
                

                input_gpuList = preprocessing(args,input_frames,frameTList,device,frameT,train=False,onlyPCA=True,model_net=model_net)
                
                B,C,T,H,W = input_frames.shape
                input_gpu = [F.interpolate(input_frames.permute(0,2,1,3,4).reshape(B*T,C,H,W), scale_factor=args.scales[0]/ (args.scales[i]),mode='bicubic', 
                    align_corners=args.align_cornerse).to(device).reshape(B,T,C,int(H*(args.scales[0]/ (args.scales[i]))),int(W*(args.scales[0]/ ( args.scales[i])))).permute(0,2,1,3,4) if(i!=0 ) else input_frames.to(device)  for i in range(args.S_tst+1)]
                

                frameTList = []
                


            t_value = Variable(t_value.to(device))
            torch.cuda.empty_cache()
            pred_starttime = time.time()
      
            pred_frameT,fine_flow = model_net(input_gpuList, t_value,normInput=[im.clone() for im in input_gpu],is_training=False,validation=validation)
            
            

            pred_time.update(time.time()-pred_starttime)
            batch_time.update(time.time() - start_time)
            
            # Calculate loss on PredFrameT and save it
            if(validation):
                rec_loss = args.rec_lambda * multi_scale_recon_loss(pred_frameT.detach().cpu().clone(),frameT.detach().cpu().clone())
                val_loss_q.append(rec_loss.item())
                
            
            if((args.dataset=="Xiph" or args.dataset=="X4K1000FPS" or args.dataset=="Inter4K88" or args.dataset=="Inter4K816") and not validation and not args.xiph2k and not args.xtest2k):
                tempsize = sizesDS[args.dataset]
                assert OH==tempsize[0] and OW==tempsize[1]

            
            pred_frameT = np.squeeze(pred_frameT.detach().cpu().numpy())[:,:OH,:OW]
            frameT = np.squeeze(frameT.detach().cpu().numpy())
            



            if args.phase != 'test_custom':
            
                """ compute PSNR & SSIM """ # From output And Ground Truth!
                output_img = np.around(denorm255_np(np.transpose(pred_frameT, [1, 2, 0])))  # [h,w,c] and [-1,1] to [0,255]
                target_img = denorm255_np(np.transpose(frameT, [1, 2, 0]))  # [h,w,c] and [-1,1] to [0,255]

            

                if(args.save_images ): 
                    testpredspath = os.path.join("log_dir",args.model_dir,args.dataset,str(testIndex//(multiple-1))) 
                    check_folder(testpredspath)
                    cv2.imwrite(os.path.join(testpredspath,str(int(t_value*multiple))+".png"),output_img)
                    cv2.imwrite(os.path.join(testpredspath,"GT"+str(int(t_value*multiple))+".png"),target_img)
                    if(testIndex % (multiple-1) == 0):
                        tempsize = [OH,OW]
                        cv2.imwrite(os.path.join(testpredspath,"0.png"),((input_frames[0,:,0,:tempsize[0],:tempsize[1]].permute(1,2,0).detach().cpu().numpy()+1)/2) * 255)
                        cv2.imwrite(os.path.join(testpredspath,str(multiple)+".png"),((input_frames[0,:,1,:tempsize[0],:tempsize[1]].permute(1,2,0).detach().cpu().numpy()+1)/2) * 255)
                        
                
                test_psnr = psnr(target_img, output_img,args)
                test_ssim = ssim_bgr(target_img, output_img)  

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




                # measure
                if(validation):
                    losses.update(rec_loss.item(), 1)
                if(not validation and (args.dataset == "Inter4K88" or args.dataset == "Inter4K816")):
                	PSNRsList[int(t_value*multiple)-1].update(test_psnr,1)
                PSNRs.update(test_psnr, 1)
                SSIMs.update(test_ssim, 1)



                if (testIndex % (multiple - 1)) == multiple - 2:
                    progress.print(testIndex)
                    if(not validation and  (args.dataset == "Inter4K88" or args.dataset == "Inter4K816")):
	                    printstring = " ".join([str(index)+": "+str(ttime.avg)+ " || " for index,ttime in enumerate(PSNRsList)])
	                    print(printstring)
                    if(args.stoptestat != -1):
                        if(testIndex > args.stoptestat):
                            break
            
        if(validation):
            print("Validation loss mean: ", torch.mean(torch.as_tensor(val_loss_q)))
        print("-----------------------------------------------------------------------------------------------")
    print("These indices were skipped: ",skipList)
    return losses.avg, PSNRs.avg, SSIMs.avg, _,PSNRsList


if __name__ == '__main__':
    main()
