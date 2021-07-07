# MHFC-FSL
This repository contains the code for our paper "MHFC: Multi-Head Feature Collaboration for Few-Shot Learning"

## Requirements

python=3.7.9

torch=1.1.0

sklearn=0.21.2

## Usage
Test
```
python main.py --mode test -g 0 --resume ckpt/miniimagenet/x_res12_best.pth.tar --r_resume ckpt/miniimagenet/r_res12_best.pth.tar --dataset miniimagenet
```

## Acknowledgments

This code is based on [ICI-FSL](https://github.com/Yikai-Wang/ICI-FSL/blob/master/V1-CVPR20/)