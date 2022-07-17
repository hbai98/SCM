# Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration
By Haotian Bai, Ruimao Zhang, Jiong Wang, Xiang Wan

This is official implementation of ["Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration"](https://github.com/164140757/SCM) in PyTorch.
Our code is developed based on https://github.com/vasgaowei/TS-CAM. Thanks for your code!

![](./figures/Intro-min.png)

## Updates
- [2022-07-10] Initial Commits. Code publically available!


## Architecture Overview
![](./figures/Arch-min.png)
## Results and Models

| Datasets | Backbone | Top1-Loc Acc | Top5-Loc Acc | GT-Known | Top1-Cls Acc | Top5-Cls Acc | Log | Checkpoints |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CUB      | Deit-S   |      76.4    |    91.6    |  96.6   |   78.5    |  94.5   |   [Google Drive](https://drive.google.com/file/d/1-Pcifaa6xNZvXG7RDD9uISZ8_XDmjVtP/view?usp=sharing)   |  [Google Drive](https://drive.google.com/drive/folders/1-FranLy5KSttCPK98ZY27TMXuriE9jkj?usp=sharing)     |
| ILSVRC   | Deit-S   |      56.1    |    66.4    |  68.8   |   76.7    |  93.0   |   [Google Drive](https://drive.google.com/file/d/1-fE8BZDvqMhjOllFyvPckELy0Jbo8A8u/view?usp=sharing)   |   [Google Drive](https://drive.google.com/drive/folders/1-HZBXo_AoK6W5gwRVh4LD8oyGDYrEc8z?usp=sharing)    |

### Visualization
![](./figures/comparison-min.png)

- Note: Image compressed for efficiency. Please refer to our paper for better visualization result.
## Usage
### Requirements
```
python==3.9
pytorch
mmcv
```
### Installation
```
conda env create -f environment.yml
conda activate SCM
```

### Inference
```
# CUB
sh val_cub.sh
# ImageNet
sh val_ilsvrc.sh
```

### Training
```
# CUB
sh train_val_cub.sh
# ImageNet
sh train_val_ilsvrc.sh
```

### Citation
If you find our paper or code useful, please consider cite our paper.
```
@inproceedings{Bai2022SCM,
  title={Weakly Supervised Object Localization via Transformer with Implicit Spatial Calibration},
  author={Haotian, Bai and Ruimao, Zhang and Jiong, Wang and Xiang, Wan},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  year={2022}
  }
```