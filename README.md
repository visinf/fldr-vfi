# Efficient Feature Extraction for High-resolution Video Frame Interpolation

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) [![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

This is the official repository accompanying the BMVC 2022 paper:

**Efficient Feature Extraction for High-resolution Video Frame Interpolation**  
Moritz Nottebaum, [Stefan Roth](https://www.visinf.tu-darmstadt.de/visinf/team_members/sroth/sroth.en.jsp) and [Simone Schaub-Mayer](https://schaubsi.github.io)  
BMVC 2022. [[paper (open access)](https://bmvc2022.mpi-inf.mpg.de/0825.pdf)] [[supplemental](https://bmvc2022.mpi-inf.mpg.de/0825_supp.zip)] [[example results](https://youtu.be/C4lgU6XXhbw)] [[talk video](https://youtu.be/wIKlm_lwf3U)] [[preprint (arXiv)](https://arxiv.org/abs/2211.14005)]

This repository contains the training and test code along with the trained weights to reproduce our results, and our test datasets Inter4K-S and Inter4K-L (subsets of [Inter4K](https://alexandrosstergiou.github.io/datasets/Inter4K/index.html)).

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
The following command will train a model from scratch:
```
python train_it.py --x_train_data_path /to/xtrain/train  --toptim
```
The flag `--x_train_data_path /to/xtrain/train` contains the location of the training data. We used [X-Train](https://github.com/JihyongOh/XVFI#X4K1000FPS) from Sim et al.
The `--toptim` flag is optional and adds a post-training optimization of the temperature parameter as described in our paper in the occlusion estimation section.

## Testing
1. Download the respective testset from the following links:

| Dataset       | Link  | 
| :---        |     :---:       | 
| X-Test     | [XVFI repository](https://github.com/JihyongOh/XVFI#X4K1000FPS)| 
| Xiph   | [Xiph benchmark](https://github.com/sniklaus/softmax-splatting/) | 
| Inter4K       | [Our subset](https://www.dropbox.com/sh/qjiht28m488u85e/AADJDwtgAP5vYIItYoFCCJkra?dl=0) ([Licence](https://github.com/alexandrosstergiou/Inter4K/blob/main/licence.txt)) | 

2. Use the file path for the test sets accordingly or update them in `main.py`:
```
--x_test_data_path /to/your/location
--xiph_data_path /to/your/location
--inter4k_data_path /to/your/location
```

3. The following line will evaluate the provided checkpoint "fLDRnet_X4K1000FPS_exp1_best_PSNR.pt" on all four testsets (X-Test, Xiph-4K, Inter4K-S, Inter4K-L): 

```
python main.py --exp_num 1 --gpu 0 --papermodel --test5scales 
```
By adding the option `--testsets` you can choose on which data you want to evaluate (options are `"Inter4K-S"`, `"Inter4k-L"`, `"X-Test"`,`"Xiph-4K"`).
The option `--papermodel` ensures all preferences are set according to the model of the paper. The option `--test5scales` adapts `args.fractions`, `args.scales`, `args.phase`, `args.S_tst` and `args.moreTstSc` to allow for additional scales for testing. 


## Acknowledgements
We thank the contributors of the following repositories for using parts of their publicly available code and 4K datasets, which were necessary to adequately train and evaluate our method:
- [XVFI](https://github.com/JihyongOh/XVFI)
- [Softmax-splatting](https://github.com/sniklaus/softmax-splatting/)
- [Inter4K](https://github.com/alexandrosstergiou/Inter4K): Our Inter4K testset for video frame interpolation is a subset of the images occuring in the Inter4K dataset shared under [CC BY-NC 4.0](https://github.com/alexandrosstergiou/Inter4K/blob/main/licence.txt).

## Citation
We hope you find our work useful. If you would like to acknowledge it in your project, please use the following citation:
```
@inproceedings{Nottebaum:2022:EFE,
  author    = {Nottebaum, Moritz and Roth, Stefan and Schaub-Meyer, Simone},
  title     = {Efficient Feature Extraction for High-resolution Video Frame Interpolation},
  booktitle = {British Machine Vision Conference  {BMVC}},
  year      = {2022}
}
```
