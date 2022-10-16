import os


befehl = []
# worst checkpoint: 24.25
befehl.append("python main.py --gpu 0 --phase 'test' --testsets \"Inter4K\" --exp_num 200 --dataset 'X4K1000FPS' --module_scale_factor 4 --S_tst 4 --multiple 8 ")
befehl.append("python main.py --gpu 0 --phase 'test' --testsets \"Inter4K\" --exp_num 200 --dataset 'X4K1000FPS' --module_scale_factor 4 --S_tst 3 --multiple 8 ")
# best one!
befehl.append("python main.py --phase 'train' --exp_num 100 --gpu 0 --dataset 'X4K1000FPS'  --S_trn 3 --S_tst 3  --pcanet --mean_vector_norm  --ds_normInput --scales 8 16 32 64 --fractions 4 16 64 256 --oneEV --ref_feat_extrac --optimizeEV   --lr_milestones 70 120 170  --ExacOneEV --takeBestModel  --allImUp ")
for i in befehl:
	os.system(i)


