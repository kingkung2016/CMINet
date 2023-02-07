# CMINet
Cross-stage Multi-scale Interaction Network for RGB-D Salient Object Detection

This is the official implementation of "Cross-stage Multi-scale Interaction Network for RGB-D Salient Object Detection" as well as the follow-ups. The paper has been published by IEEE Signal Processing Letters, 2022. The paper link is [here](https://ieeexplore.ieee.org/document/9956739).
****

## Content
* [Run CMINet code](#Run-CMINet-code)
* [Pretrained models](#Pretrained-models)
* [Saliency maps](#Saliency-maps)
* [Evaluation tools](#Evaluation-tools)
* [Citation](#Citation)
****

## Run CMINet code
- Train <br>
  run `python train.py` <br>
  \# put pretrained models in the pretrained folder <br>
  \# set '--train-root' to your training dataset folder
  
- Test <br>
  run `python test.py` <br>
  \# set '--test-root' to your test dataset folder <br>
  \# set '--ckpt' as the correct checkpoint <br>
****

## Pretrained models
  - The pretrained models can be downloaded in [Baidu Cloud](https://pan.baidu.com/s/1SXAC1DtgeuyQ_WxlyI9VeQ) (fetach code is moad). Then put the pretrained models such as 'resnet_50.pth' in the pretrained folder.
****

## Saliency maps
  - The saliency maps can be approached in [Baidu Cloud](https://pan.baidu.com/s/1SXAC1DtgeuyQ_WxlyI9VeQ) (fetach code is moad). Note that all testing results are provided not only including those listed in the paper.
****

## Evaluation tools
- The evaluation tools, training and test datasets can be obtained in [RGBD-SOD-tools](https://github.com/kingkung2016/RGBD-SOD-tools).
****

## Citation
```
@ARTICLE{yi2022cross,
  author={Yi, Kang and Zhu, Jinchao and Guo, Fu and Xu, Jing},
  journal={IEEE Signal Processing Letters}, 
  title={Cross-Stage Multi-Scale Interaction Network for RGB-D Salient Object Detection}, 
  year={2022},
  volume={29},
  number={},
  pages={2402-2406},
  doi={10.1109/LSP.2022.3223599}
}

```
****


