import os


befehl = []
# worst checkpoint: 24.25

befehl.append("python main.py --phase 'train' --exp_num 2060 --gpu 0 --dataset 'X4K1000FPS'  --S_trn 3 --S_tst 3  --pcanet --mean_vector_norm  --ds_normInput --scales 8 16 32 64 --fractions 4 16 64 256 --oneEV --ref_feat_extrac --optimizeEV   --lr_milestones 70 120 170  --ExacOneEV --takeBestModel  --allImUp --softsplat    --sminterp  --ownsmooth  --noResidAddup --impmasksoftsplat --cutoffUnnec --fixsmoothtwistup --sminterpInpIm  --patch_size 512 --tempAdamfix --continue_training")
# best one!
befehl.append("python main.py --phase 'train' --exp_num 4002 --gpu 0 --dataset 'X4K1000FPS'  --S_trn 3 --S_tst 3  --pcanet --mean_vector_norm  --ds_normInput --scales 8 16 32 64 --fractions 4 16 64 256 --oneEV --ref_feat_extrac --optimizeEV   --lr_milestones 70 120 170  --ExacOneEV --takeBestModel  --allImUp --softsplat  --forwendflowloss --warp_alpha 0.05  --sminterp  --ownsmooth  --noResidAddup --impmasksoftsplat --cutoffUnnec --fixsmoothtwistup --sminterpInpIm  --interpOrigForw --patch_size 512")

for i in befehl:
	os.system(i)


