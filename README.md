# MHFC-FSL
This repository contains the code for our paper "MHFC: Multi-Head Feature Collaboration for Few-Shot Learning".

Note that, our inductive setting is the "non-standardized inductive setting", which adopts the query feature when reducing the feature's dimension.

## Requirements

python=3.7.9

torch=1.1.0

sklearn=0.21.2

## Usage
Test
```
python main.py --mode test -g 0 r --dataset miniimagenet
```


## Acknowledgments

This code is based on [ICI-FSL](https://github.com/Yikai-Wang/ICI-FSL/blob/master/V1-CVPR20/)

## Citation

If you found the provided code useful, please cite our work.

```
@inproceedings{shao2021mhfc,
  title={Mhfc: Multi-head feature collaboration for few-shot learning},
  author={Shao, Shuai and Xing, Lei and Wang, Yan and Xu, Rui and Zhao, Chunyan and  Wang, Yan-Jiang and Liu, Bao-Di},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}
```