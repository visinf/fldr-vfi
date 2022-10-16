frameTTemp = np.copy(frameT)
            input_framesTemp= np.copy(input_frames)
            # Convert Input and Groundtruth to DCT
            for moX in range(input_frames.shape[1]):
                for moI in range(input_frames.shape[0]): # 8 batchsize
                    frameT[moI,moX,:,:] = scF.dctn(frameT[moI,moX,:,:])
                    for moV in range(input_frames.shape[2]):
                        input_frames[moI,moX,moV,:,:] = scF.dctn(input_frames[moI,moX,moV,:,:])



def fourier_trans(i,padded_fourier,wiS,components_used,pca_reload,rounding,data_saved):
    non_zeros = 0
    # mean und std vom bild normalisieren!!!!
    origim = i
    
    for k in range(origim.shape[0]//wiS):
        for l in range(origim.shape[1]//wiS):
            padded_fourier[k*wiS:(k+1)*wiS,l*wiS:(l+1)*wiS] = scipy.fft.dct(origim[k*wiS:(k+1)*wiS,l*wiS:(l+1)*wiS])
            #data_saved.add()
            pca_transformed = pca_reload.transform(padded_fourier[k*wiS:(k+1)*wiS,l*wiS:(l+1)*wiS].reshape(1,-1))
            pca_transformed[:,components_used:] = 0
            # Rounding step:
            if(rounding > 1):
                pca_transformed = (pca_transformed/rounding).astype(int)*rounding

            non_zeros +=  np.count_nonzero(pca_transformed)

            #data_saved.add(pca_transformed)
            back_transformed = pca_reload.inverse_transform(pca_transformed).reshape(wiS,wiS)
            # ACTUAL JPEG!!
            #back_transformed = (padded_fourier[index][k*wiS:(k+1)*wiS,l*wiS:(l+1)*wiS]/quantization_table()).astype(int)*quantization_table()


            padded_fourier[k*wiS:(k+1)*wiS,l*wiS:(l+1)*wiS] = scipy.fft.idct(back_transformed)
    
    return padded_fourier,non_zeros,data_saved



# SKCUDA: TO PCA and Back and to Ortsraum
    #X_gpu = gpuarray.GPUArray(pca_ready.shape, np.float64, order="F")
    #X_gpu.set(pca_ready)
    
    #pca = cuPCA(n_components=components_used)
    #pca_transformed = pca.fit_transform(X_gpu)
    #pca_back = pca.inverse_transform(pca_transformed)
    # from pca DS to numpy
    #pca_back.get(pca_ready)
    

def train(model_net, criterion, device, save_manager, args):
    SM = save_manager
    multi_scale_recon_loss = criterion[0]
    smoothness_loss = criterion[1]

    optimizer = optim.Adam(model_net.parameters(), lr=args.init_lr, betas=(0.9, 0.999),
                           weight_decay=args.weight_decay)  # optimizer
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_dec_fac)

    last_epoch = 0
    best_PSNR = 0.0

    # Here ---LATEST----- is loaded
    if args.continue_training:
        checkpoint = SM.load_model()
        last_epoch = checkpoint['last_epoch'] + 1
        best_PSNR = checkpoint['best_PSNR']
        model_net.load_state_dict(checkpoint['state_dict_Model'])
        optimizer.load_state_dict(checkpoint['state_dict_Optimizer'])
        scheduler.load_state_dict(checkpoint['state_dict_Scheduler'])
        print("Optimizer and Scheduler have been reloaded. ")
    scheduler.milestones = Counter(args.lr_milestones)
    scheduler.gamma = args.lr_dec_fac
    print("scheduler.milestones : {}, scheduler.gamma : {}".format(scheduler.milestones, scheduler.gamma))
    start_epoch = last_epoch

    # switch to train mode
    model_net.train()

    start_time = time.time()

    SM.write_info('Epoch\ttrainLoss\ttestPSNR\tbest_PSNR\n')
    print("[*] Training starts")

    # Main training loop for total epochs (start from 'epoch=0')
    valid_loader = get_test_data(args, multiple=4, validation=True)  # multiple is only used for X4K1000FPS

    params = DCTParams(20,args.gpu,1,1)

    for epoch in range(start_epoch, args.epochs):
        train_loader,psnr_diff = get_train_data(args,
                                      max_t_step_size=32,device=device)  # max_t_step_size (temporal distance) is only used for X4K1000FPS

        batch_time = AverageClass('batch_time[s]:', ':6.3f')
        losses = AverageClass('Loss:', ':.4e')
        progress = ProgressMeter(len(train_loader), batch_time, losses, prefix="Epoch: [{}]".format(epoch))

        print('Start epoch {} at [{:s}], learning rate : [{}]'.format(epoch, (str(datetime.now())[:-7]),
                                                                      optimizer.param_groups[0]['lr']))

        # train for one epoch
        for trainIndex, (frames, t_value) in enumerate(train_loader):

            input_frames = frames[:, :, :-1, :] # [B, C, T, H, W]  # 8,3,2,384,384
            frameT = frames[:, :, -1, :]  # [B, C, H, W]     # 8,3,384,384
            shap = input_frames.shape
            


            #[chan * components,blocks_y,blocks_x]
            if(args.dctnet):
                input_frames = input_frames.reshape(shap[0],-1,shap[3],shap[4])
                shap = input_frames.shape # overwrites shap here
                pcas = [0 for i in range(shap[0])]
                # TODO: doppelerstellung hier, übergebe input_gpu bei to_dctpca stattdessen
                input_gpu = torch.zeros(shap,device=device)
                for i in range(shap[0]):
                    # DCTPCA
                    input_gpu[i,:],pcas[i] = torch.as_tensor(to_dctpca(input_frames[i,:,:,:],params),device=device)
                    # DCT
                    frameT[i,:] = torch.from_numpy(to_dct(frameT[i,:].numpy()))
            else:
                input_gpu = input_frames.to(device) # as to_dctpca already brings it on GPU

            frameT = frameT.to(device)  # ground truth for frameT
            t_value = t_value.to(device)  # [B,1]


            optimizer.zero_grad()
            

            if(args.dctnet):    ## DCTNET
                pred_frameT_pyramid, pred_flow_pyramid, occ_map, simple_mean = model_net(input_gpu, t_value,pcas)
            else:               ## XVFINET
                pred_frameT_pyramid, pred_flow_pyramid, occ_map, simple_mean = model_net(input_gpu, t_value)


            rec_loss = 0.0
            smooth_loss = 0.0
            for l, pred_frameT_l in enumerate(pred_frameT_pyramid):
                temp = l
                if(args.no_ds_rfb):
                    temp = 0
                rec_loss += args.rec_lambda * multi_scale_recon_loss(pred_frameT_l,
                                                                     F.interpolate(frameT, scale_factor=1 / (2 ** temp),
                                                                                   mode='bicubic', align_corners=False))


            smooth_loss += 0.5 * smoothness_loss(pred_flow_pyramid[0],
                                                 F.interpolate(frameT, scale_factor=1 / args.module_scale_factor,
                                                               mode='bicubic',
                                                               align_corners=False))  # Apply 1st order edge-aware smoothness loss to the fineset level

            rec_loss /= len(pred_frameT_pyramid)
            pred_frameT = pred_frameT_pyramid[0]  # final result I^0_t at original scale (s=0)
            
            temp = args.S_trn
            if(args.no_ds_rfb):
                temp = 0
            pred_coarse_flow = 2 ** (temp) * F.interpolate(pred_flow_pyramid[-1], scale_factor=2 ** (
                temp) * args.module_scale_factor, mode='bicubic', align_corners=False)

            pred_fine_flow = F.interpolate(pred_flow_pyramid[0], scale_factor=args.module_scale_factor, mode='bicubic',
                                           align_corners=False)

            total_loss = rec_loss + smooth_loss

            # compute gradient and do SGD step
            total_loss.backward()  # Backpropagate
            optimizer.step()  # Optimizer update

            # measure accumulated time and update average "batch" time consumptions via "AverageClass"
            # update average values via "AverageClass"
            losses.update(total_loss.item(), 1)
            batch_time.update(time.time() - start_time)
            start_time = time.time()

            if trainIndex % args.freq_display == 0:
                ####        Backtransform it!!!!!!!!!!!!!       #######################
                #pred_frameT = torch.from_numpy(scF.idctn(scaleFrameT.backscale(pred_frameT).detach().cpu().numpy()))
                #frameT = torch.from_numpy(scF.idctn(scaleFrameT.backscale(frameT).detach().cpu().numpy()))

                progress.print(trainIndex)
                batch_images = get_batch_images(args, save_img_num=args.save_img_num,
                                                save_images=[pred_frameT, pred_coarse_flow, pred_fine_flow, frameT,
                                                             simple_mean, occ_map])
                ########### COMMENTED THIS LINE OUT #################################################################
                #cv2.imwrite(os.path.join(args.log_dir, '{:03d}_{:04d}_training.png'.format(epoch, trainIndex)), batch_images)
        
        # Error of bil and DCT pictures
        print(psnr_diff[0].avg)
        print(psnr_diff[1].avg)

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
            'state_dict_Model': model_net.state_dict(),
            'state_dict_Optimizer': optimizer.state_dict(),
            'state_dict_Scheduler': scheduler.state_dict()}
            SM.save_best_model(combined_state_dict, False)    
            continue

        #### EVALUATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # if (epoch + 1) % 10 == 0 or epoch==0:
        val_multiple = 4 if args.dataset == 'X4K1000FPS' else 2
        print('\nEvaluate on test set (validation while training) with multiple = {}'.format(val_multiple))
        postfix = '_val_' + str(val_multiple) + '_S_tst' + str(args.S_tst)
        testLoss, testPSNR, testSSIM, final_pred_save_path = test(valid_loader, model_net, criterion, epoch, args,
                                                                  device, multiple=val_multiple, postfix=postfix,
                                                                  validation=True)

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
            'state_dict_Model': model_net.state_dict(),
            'state_dict_Optimizer': optimizer.state_dict(),
            'state_dict_Scheduler': scheduler.state_dict()}

        # Saves the LATEST model!! If best model, it overwrites current best model!
        SM.save_best_model(combined_state_dict, best_PSNR_flag)

        # EPOCH SAVES
        if (epoch + 1) % 10 == 0:
            SM.save_epc_model(combined_state_dict, epoch)
        SM.write_info('{}\t{:.4}\t{:.4}\t{:.4}\n'.format(epoch, losses.avg, testPSNR, best_PSNR))

    # Until here, everything is done for each epoch
    print("------------------------- Training has been ended. -------------------------\n")
    print("information of model:", args.model_dir)
    print("best_PSNR of model:", best_PSNR)





if(not self.args.no_refine):
            featAndWarped = self.channredux(torch.cat([feat0_l, feat1_l, warped0_l, warped1_l],dim=1))
            image_reduxed = self.channredux_image(torch.cat([warped_img0_l, warped_img1_l],dim=1))

            refine_out = self.refine_unet(torch.cat([featAndWarped, x_l[:,:,0,:,:], x_l[:,:,1,:,:],image_reduxed , flow_t0_l, flow_t1_l],dim=1))
            #refine_out = self.refine_unet(torch.cat([F.pixel_shuffle(torch.cat([feat0_l, feat1_l, warped0_l, warped1_l],dim=1), self.scale), x_l[:,:,0,:,:], x_l[:,:,1,:,:], warped_img0_l, warped_img1_l, flow_t0_l, flow_t1_l],dim=1))
            
            #print(refine_out.shape)

            occ_0_l = torch.sigmoid(refine_out[:, 0:1, :, :])
            occ_1_l = 1-occ_0_l
            
            out_l = (1-t_value)*occ_0_l*warped_img0_l + t_value*occ_1_l*warped_img1_l
            out_l = out_l / ( (1-t_value)*occ_0_l + t_value*occ_1_l ) + refine_out[:, 1:151, :, :]

            #print("out_l: ",out_l.shape)
            end_dct = []
            for index,i in enumerate(pcas):
                end_dct.append(from_dctpca_to_dct_diff(out_l[index,:].unsqueeze(0),params,i))#torch.as_tensor(i.eigenvectors,device=self.device),torch.as_tensor(i.mean,device=self.device),

            out_l = torch.cat(end_dct,dim=0)
            #print("Endresult shape: ",out_l.shape)






def to_dct(im,wiS):
    
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

    # Padding
    image = np.zeros((chan,height + pad_y, width + pad_x))
    #image[:,:height,:width] = im[:,:,:]
    
    image[:,:,:] = np.pad(im,((0,0),(0,pad_y),(0,pad_x)) ,mode="reflect")
    
    # Into blocks
    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    

    blocked = scF.dctn(blocked,axes=(3,4))    #blocked_gpu = cufft.fftn(blocked_gpu,axes=(3,4)).real 

    # to C,H,W
    #blocked = np.transpose(fourier_back,(0,1,2,3,4)) # [C,blockx,blocky,wiS,wiS]
    blocked = blocked.reshape(chan,blocks_x,wiS*blocks_y,wiS)
    blocked = np.concatenate([ blocked[:,i,:,:]for i in range(blocked.shape[1])],axis=2)
    
    #ret_im = blocked[:chan,:height,:width]

    return blocked #clipping(ret_im)



# ret:  [chan * components,blocks_y,blocks_x]
def to_dctpca(im,params,components_fraction=0):
    #print(im.shape)
    

    cp.cuda.Device(params.gpu).use()
    wiS = params.wiS
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

    #print("My big blocked shape BEFORE!!!!!!!!: ",image.shape)
    # Into blocks
    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    #print("My big blocked shape!!!!!!!!: ",blocked.shape)


    

    #CUPY FFT
    blocked = scF.dctn(blocked,axes=(3,4))    #blocked_gpu = cufft.fftn(blocked_gpu,axes=(3,4)).real 
    pca_ready = blocked.reshape(-1,wiS*wiS)


    # GPU########################################
    #pca_ready_gpu = cp.asarray(pca_ready)
    pca_ready_gpu = torch.as_tensor(pca_ready,dtype=torch.float64,device=params.gpu)


    choice = False
    if(not choice):
        pca = MYPCA(n_components=components_used)
        res = pca.fit_transform(pca_ready_gpu)
    else:
        #rng = cp.random.default_rng()
        #partofit = cp.random.choice(cp.arange(pca_ready_gpu.shape[0]),int(pca_ready_gpu.shape[0]*data_used),replace=False)
        partofit = torch.randperm(pca_ready_gpu.shape[0])[:int(pca_ready_gpu.shape[0]*data_used)]
        partofit = pca_ready_gpu[partofit,:]
        print(partofit.shape)
        pca = MYPCA(n_components=components_used)
        pca.fit(partofit)
        print(pca_ready_gpu.shape)
        res = pca.transform(pca_ready_gpu)
    
    #print(chan,blocks_y,blocks_x,components_used)
    res = res.reshape(chan,blocks_y,blocks_x,components_used)
    res = res.permute((0,3,1,2))#cp.transpose(res,(0,3,1,2))
    res = res.reshape(-1,blocks_y,blocks_x) # chan * components,blocks_y,blocks_x
    
    # Normalize it!
    mi = torch.min(res)
    ma = torch.max(res)


    res = (((res - mi)/(ma-mi))*2)-1


    

    mi = torch.as_tensor(mi,device=params.gpu)
    ma = torch.as_tensor(ma,device=params.gpu)
    #pca.store_sth((torch.as_tensor(mi),torch.as_tensor(ma)),"mima")
    pca.store_sth((mi,ma),"mima")

    
    
    return res,pca

# [3,2160,4096]
def dct_inverse(im,wiS_param=0,params=0):
    #print(params)
    if(wiS_param == 0):
        wiS = params.wiS
    else:
        wiS = wiS_param
    #print("wiS: ",wiS)
    chan = im.shape[0]
    height = im.shape[1]
    width = im.shape[2]
    
    image = np.zeros((chan,height, width))
    
    # If this line is missing, usually an error occurs
    image[:,:,:] = im[:,:,:]
    
    #print(pad_x,pad_y)
    blocks_y = height//wiS #+(1 if(pad_y >0)else 0)
    blocks_x = width//wiS #+(1 if(pad_x >0)else 0)

    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    
    blocked = scF.idctn(blocked,axes=(3,4))
    
    blocked = np.transpose(blocked,(0,1,2,3,4)) # [C,blockx,blocky,wiS,wiS]
    blocked = blocked.reshape(chan,blocks_x,wiS*blocks_y,wiS)
    blocked = np.concatenate([ blocked[:,i,:,:]for i in range(blocked.shape[1])],axis=2)
    
    blocked = blocked[:,:params.h,:params.w]

    return clipping(blocked) #blocked

#res: [chan * components,blocks_y,blocks_x] (tensor)
# eigenvectors (tensor)
# mean (tensor)
# input: [8,150,20,20]
# eigenvectors: [400,400]
# mean: [400]
def from_dctpca_to_dct_diff(res,params,pca,cut_back=False,no_dct=False):
    #print(eigenvectors.shape)
    B,comps,BY,BX = res.shape
    eigenvectors = torch.as_tensor(pca.eigenvectors,device=params.gpu)
    mean = torch.as_tensor(pca.mean,device=params.gpu)
    wiS = params.wiS
    sized = wiS**2
    #print("mean: ",mean.shape)
    temp = res.permute((0,2,3,1))
    temp = temp.reshape(-1,comps)
    temp = temp.reshape(-1,3,comps//3)
    temp = temp.reshape(-1,comps//3).double()

    # back unnormalization          res = (((res - mi)/(ma-mi))*2)-1
    #mi,ma = pca.store["mima"]
    #mi = torch.as_tensor(mi)
    #ma = torch.as_tensor(ma)

    #temp = ((temp+1)/2)*(ma - mi) + mi

    # Components to actual DCT
    back = torch.matmul(temp,eigenvectors[:comps//3,:])
    back = back.float() # [n,400]
    
    back = back + mean.float()

    # Back transformation to Right format
    back = back.reshape(-1,3,sized).reshape(B,BX,BY,3,sized).permute((0,3,1,2,4)).reshape(B,3,BX,BY,wiS,wiS) # now: [C,blockx,blocky,wiS,wiS]
    back = back.reshape(B,3,BX,wiS*BY,wiS)
    back = torch.cat([ back[:,:,i,:,:]for i in range(back.shape[2])],axis=3)
    if(cut_back):
        back = back[:,:,:params.h,:params.w]

    print("min max: "  ,torch.min(back),torch.max(back))

    return back
# [C,blockx,blocky,wiS,wiS]
    #blocked = blocked.reshape(chan,blocks_x,wiS*blocks_y,wiS)
    #blocked = np.concatenate([ blocked[:,i,:,:]for i in range(blocked.shape[1])],axis=2)







def pca_trans_and_back_cupy(im,components_used,wiS,data_used=1,device=0):
    cp.cuda.Device(device).use()

    #x = cp.array([1, 2, 3, 4, 5])
    #print("Device: ",x.device)
    #wiS = props.wiS
    #components_used = props.components_used
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
    #print(height,width,chan,im.shape)
    
    # Cut image
    image[:,:height,:width] = im[:,:,:]
 
    # Into blocks
    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    


    #CUPY FFT
    blocked = scF.dctn(blocked,axes=(3,4))    #blocked_gpu = cufft.fftn(blocked_gpu,axes=(3,4)).real 


    pca_ready = blocked.reshape(-1,wiS*wiS)
    pca_ready_gpu = cp.asarray(pca_ready)


    pca = MYPCA(n_components=components_used)
    res = pca.fit_transform(pca_ready_gpu)
    #print("transform result shape: ",res.shape)
    returned =pca.inverse_transform(res)

    pca_back = returned.reshape(chan,blocks_y,blocks_x,wiS,wiS)
    pca_back = cp.asnumpy(pca_back)
    fourier_back = scF.idctn(pca_back,axes=(3,4))#cufft.ifftn(pca_back,axes=(3,4)).real

    
    # to C,H,W
    blocked = np.transpose(fourier_back,(0,1,2,3,4)) # [C,blockx,blocky,wiS,wiS]
    blocked = blocked.reshape(chan,blocks_x,wiS*blocks_y,wiS)
    blocked = np.concatenate([ blocked[:,i,:,:]for i in range(blocked.shape[1])],axis=2)
    
    ret_im = blocked[:chan,:height,:width]

    return clipping(ret_im)


# image: [C, H, W] [0,1]
# ret: [C, H, W] [0,1]
def pca_trans_and_back(im,components_used,wiS,data_used=1):
    print(components_used,wiS,im.shape)
    #wiS = props.wiS
    #components_used = props.components_used
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
    #print(height,width,chan,im.shape)
    
    # Cut image
    image[:,:height,:width] = im[:,:,:]
    
    # Into blocks
    blocked = as_strided(image,shape=(chan,blocks_x,blocks_y,wiS,wiS),strides=(8*prod(image.shape[1:]),wiS*8,blocks_x*wiS*8*wiS, wiS*blocks_x*8,8))
    # Make DCT
    blocked = scF.dctn(blocked,axes=(3,4),type=1) #scF.fftn(blocked,axes=(3,4)).real   
    pca_ready = blocked.reshape(-1,wiS*wiS)
    # TO PCA and Back and to Ortsraum
    pca = PCA(n_components=components_used)
    pca_transformed = pca.fit_transform(pca_ready)
    #pca_transformed[:,components_used:] = 0
    pca_back = pca.inverse_transform(pca_transformed)
    pca_back = pca_back.reshape(chan,blocks_y,blocks_x,wiS,wiS)
    pca_back = scF.idctn(pca_back,axes=(3,4),type=1) #scF.ifftn(pca_back,axes=(3,4)).real 
    
    # to C,H,W
    blocked = np.transpose(pca_back,(0,1,2,3,4)) # [C,blockx,blocky,wiS,wiS]
    blocked = blocked.reshape(chan,blocks_x,wiS*blocks_y,wiS)
    blocked = np.concatenate([ blocked[:,i,:,:]for i in range(blocked.shape[1])],axis=2)
    
    ret_im = blocked[:chan,:height,:width]

    return clipping(ret_im)
