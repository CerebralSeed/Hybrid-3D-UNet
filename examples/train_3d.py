import torch
from hybrid3d_unet_pytorch.diffsubmodels import Unet2d, Unet3d
from hybrid3d_unet_pytorch.diffusion import GaussianDiffusion, Trainer


device=torch.device("cuda:0")

unetmodel = Unet3d(24)

print(sum(p.numel() for p in unetmodel.parameters()))
model=GaussianDiffusion(unetmodel, image_size=64)
model.to(device)
folder ="Hands/Hands_New/"
batch_size=32

trainer = Trainer(model, folder,device=device, train_batch_size=batch_size,
                  fp16=False, amp=False, train_num_steps=500000, train_lr=0.002,
                  ema_decay=0.995, ema_update_every=5, save_and_sample_every=5000,
                  tag='', results_folder='./results-hands-3d-v1')

trainer.train()
"""
CITATIONS:
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
"""
