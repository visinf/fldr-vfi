# fldr-vfi
This is the official repository accompanying the BMVC 2022 paper:
M. Nottebaum, S. Roth and S. Schaub-Meyer. **Efficient Feature Extraction for High-resolution Video Frame Interpolation** (BMVC 2022).

[Paper]() | [Preprint (arXiv)]() | [Video]()

The repository contains:
- The training and testing code for our approach
- The checkpoint of the paper model
- Links to all necessary ressources to do training and testing


## Testing
In order to evaluate the model on X-Test, Inter4K-S, Inter4K-L and Xiph you need to first fetch the respective testset. The following table contains links where you can find them.

| Dataset       | Link  | 
| :---        |     :---:       | 
| X-Test     | [XVFI repo](https://github.com/JihyongOh/XVFI)| 
| Xiph   | [Xiph generation](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark.py) | 
| Inter4k       | [our link(TODO)]() | 

After downloading the testsets you can configure the testset location for the program by using the following flags for execution:
```
--x_test_data_path /to/your/location
--xiph_data_path /to/your/location
--inter4k_data_path /to/your/location
```
You can also simply change the default values of the flags in `main.py`
After that, you need to put the checkpoint "fLDRnet_X4K1000FPS.pt" into `git_dir/checkpoint_dir/fLDRnet_X4K1000FPS_exp1/`, if not already there.
If you now run 
```
python main.py --exp_num 1 --gpu 0 --papermodel --test5scales 
```
, the model will automatically test on all four testsets.
By adding the option `--testsets` you can choose on which data you want to evaluate (options are `"Inter4K88"`, `"Inter4k816"`, `"X4K1000FPS"`,`"Xiph"`).
The option `--papermodel` ensures all preferences are set according to the model of the paper. The option `--test5scales` adapts `args.fractions`,`args.scales`,`args.phase`,`args.S_tst` and `args.moreTstSc` to allow for additional scales for testing. 

## Training
In order to train a model from scratch onto the [X-Train dataset](https://github.com/JihyongOh/XVFI), you need to first determine which unique experiment number (exp_num) you give it. Additionally you need to specify the location of X-Train, by passing the `--x_train_data_path /to/xtrain/train` flag for the following training command:
```
python main.py --phase 'train' --exp_num 2 --gpu 0 --papermodel --x_train_data_path /to/xtrain/train
```
After training is done (200 epochs), you can optionally do the T-optimization. For this delete the `_latest.pt` checkpoint from the directory `checkpoint_dir/fLDRnet_X4K1000FPS_exp2/`. Then use your previous train command and add the following options to it `--epochs 220 --TOptimization --sminterpWT --init_lr 0.001`.
Now the current T value is printed during this postoptimization. You can stop training when T is converging.

## Acknowledgements
We thank [Sim et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Sim_XVFI_eXtreme_Video_Frame_Interpolation_ICCV_2021_paper.pdf), [Niklaus et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf) and [Stergiou and Poppe](https://arxiv.org/pdf/2111.00772.pdf) for providing 4K datasets, which were necessary to adequately evaluate and train our method.

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
TODO
```




