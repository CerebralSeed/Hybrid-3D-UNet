# Hybrid-3D-UNet
#### Model for Hybrid 3D UNet
Welcome to the Github for Hybrid 3d UNet for PyTorch! This makes some modifications to the UNet to allow for 

### Article:
https://jjohnson-777.medium.com/fixing-stable-diffusions-achilles-heel-43da2563647e

### Usage:
1. First install Hybrid 3d UNet for PyTorch Git Repo with: 

```
pip install git+https://github.com/CerebralSeed/Hybrid-3D-UNet.git#egg=hybrid3dunet
```

2. Scripts ready to run each model are in the `/examples` folder. Update the folder for images you'd like to train on and adjust any other arguments to your liking.

### Using Hybrid 3d UNet Modules in your Models
After installing Hybrid 3d UNet for Pytorch, you can use the modules in your PyTorch models to transition from 1d to 2d and vice versa, or 2d to 3d and vice versa via:
```python
import torch
from hybrid3dunet.diffsubmodules import Conv2dtoConv1d, Conv1dtoConv2d, Conv2dtoConv3d, Conv3dtoConv2d

updim1d2d = Conv1dtoConv2d(in_channels=3, out_channels=16, depth_dim=32, kernel=3, bias=False)

dummy_tensor = torch.rand(5, 3, 32)

output = updim1d2d(dummy_tensor)

print(output.size())

updim2d3d = Conv2dtoConv3d(in_channels=16, out_channels=16, depth_dim=32, kernel=3, bias=False)

output = updim2d3d(output)

print(output.size())

downdim3d2d = Conv3dtoConv2d(in_channels=16, out_channels=16, depth_dim=32, kernel=None,
                             bias=False)  # kernel defaults to size of 1

output = downdim3d2d(output)

print(output.size())

downdim2d1d = Conv2dtoConv1d(in_channels=16, out_channels=3, depth_dim=32, kernel=None,
                             bias=False)

output = downdim2d1d(output)

print(output.size())

```

```python
torch.Size([5, 16, 32, 32])
torch.Size([5, 16, 32, 32, 32])
torch.Size([5, 16, 32, 32])
torch.Size([5, 3, 32])
```

### Model Weights: 

https://drive.google.com/drive/folders/1-TbO244Ilgc2-V5S4SlvImWi8HDCzpae?usp=sharing

### Results:

![alt text](https://github.com/CerebralSeed/Hybrid-3D-UNet/blob/main/compare-chart.jpg?raw=true)


### 25 Random Samples after 100 Epochs:

#### UNet (2D)

![alt text](https://github.com/CerebralSeed/Hybrid-3D-UNet/blob/main/2d_train_results/sample-100-loss-0.07358751328475774.png?raw=true)

#### UNet 3D Hybrid

![alt text](https://github.com/CerebralSeed/Hybrid-3D-UNet/blob/main/3d_train_results/sample-100-loss-0.05175705623235553.png)

(Samples at the end of each epoch are in folders 2d_train_results and 3d_train_results)


### Citations

1) Dataset from: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009. (CIFAR10)
2) Diffusion Model, trainer and Unet modified from https://github.com/lucidrains/denoising-diffusion-pytorch
3) Efficient Attention modified from https://github.com/cmsflash/efficient-attention with paper:
4) @inproceedings{shen2021efficient,
    author = {Zhuoran Shen and Mingyuan Zhang and Haiyu Zhao and Shuai Yi and Hongsheng Li},
    title = {Efficient Attention: Attention with Linear Complexities},
    booktitle = {WACV},
    year = {2021},
}
5) @article{afifi201911kHands,
title = {11K Hands: gender recognition and biometric identification using a large dataset of hand images},
author = {Afifi, Mahmoud},
journal = {Multimedia Tools and Applications},
doi = {10.1007/s11042-019-7424-8},
url = {https://doi.org/10.1007/s11042-019-7424-8},
year={2019}
}
6) Nuzzi, Cristina; Pasinetti, Simone; Pagani, Roberto; Coffetti, Gabriele; Sansoni, Giovanna (2021), 
“HANDS: a dataset of static Hand-Gestures for Human-Robot Interaction”, 
Mendeley Data, V1, doi: 10.17632/ndrczc35bt.1
}
7) @article{tompson14tog,
  author = {Jonathan Tompson and Murphy Stein and Yann Lecun and Ken Perlin}
  title = {Real-Time Continuous Pose Recovery of Human Hands Using Convolutional Networks,
  journal = {ACM Transactions on Graphics},
  year = {2014},
  month = {August},
  volume = {33}
}
8) https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/
9) https://www.kaggle.com/grassknoted/asl-alphabet
