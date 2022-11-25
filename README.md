# fldr-vfi
This is the official repository accompanying the BMVC 2022 paper:

**Efficient Feature Extraction for High-resolution Video Frame Interpolation**  
M. Nottebaum, S. Roth and S. Schaub-Meyer  
BMVC 2022

[Paper]() | [Preprint (arXiv)]() | [Video]()

The repository contains:
- The training and testing code of our approach
- The checkpoint of our video frame interpolation model to reproduce the paper results
- Links to all testsets, including Inter

## Installation
The following steps will set up a local copy of the repository.
1. Create conda environment:
```
conda create --name fldrnet
conda activate fldrnet
```
2. Install PyTorch:
```
conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```
3. Install the dependencies:
```
pip install -r requirements.txt
```

## Training
Coming soon

## Testing
1. Download the respective testset:

| Dataset       | Link  | 
| :---        |     :---:       | 
| X-Test     | [XVFI repo](https://github.com/JihyongOh/XVFI)| 
| Xiph   | [Xiph generation](https://github.com/sniklaus/softmax-splatting/blob/master/benchmark.py) | 
| Inter4K       | [Our test images (Coming soon)]() | 

2. Use the file path for the test sets accordingly or update them in `main.py`:
```
--x_test_data_path /to/your/location
--xiph_data_path /to/your/location
--inter4k_data_path /to/your/location
```

3. The following line will evaluate the provided checkpoint "fLDRnet_X4K1000FPS.pt" on all four testsets (X-Test, Xiph, Inter4K-S, Inter4K-L): 

```
python main.py --exp_num 1 --gpu 0 --papermodel --test5scales 
```
By adding the option `--testsets` you can choose on which data you want to evaluate (options are `"Inter4K88"`, `"Inter4k816"`, `"X4K1000FPS"`,`"Xiph"`).
The option `--papermodel` ensures all preferences are set according to the model of the paper. The option `--test5scales` adapts `args.fractions`,`args.scales`,`args.phase`,`args.S_tst` and `args.moreTstSc` to allow for additional scales for testing. 


## Acknowledgements
We thank [Sim et al.](https://openaccess.thecvf.com/content/ICCV2021/papers/Sim_XVFI_eXtreme_Video_Frame_Interpolation_ICCV_2021_paper.pdf), [Niklaus et al.](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf) and [Stergiou and Poppe](https://arxiv.org/pdf/2111.00772.pdf) for providing 4K datasets, which were necessary to adequately evaluate and train our method.

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
@inproceedings{Nottebaum:2022:EFE,
  author    = {Nottebaum, Moritz and Roth, Stefan and Schaub-Meyer, Simone},
  title     = {Efficient Feature Extraction for High-resolution Video Frame Interpolation},
  booktitle = {British Machine Vision Conference  {BMVC}},
  publisher = {{BMVA} Press},
  year      = {2022}}
```
