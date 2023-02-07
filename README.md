# CMINet
Cross-stage Multi-scale Interaction Network for RGB-D Salient Object Detection

This is the official implementation of "Cross-stage Multi-scale Interaction Network for RGB-D Salient Object Detection" as well as the follow-ups. The paper has been published by IEEE Signal Processing Letters, 2022. The paper link is [here](https://ieeexplore.ieee.org/document/9956739).
****

## Content
* [Run CMINet code](#Run-CMINet-code)
* [Pretained models](#Pretained-models)
* [Saliency maps](#Saliency-maps)
* [Evaluation tools](#Evaluation-tools)
* [Citation](#Citation)
****

## Run MoADNet code
- Train <br>
  run `python train.py` <br>
  \# put pretrained models to the pretrained folder
  \# set '--train-root' to your training dataset folder
  
- Test <br>
  run `python test.py` <br>
  \# set '--test-root' to your test dataset folder <br>
  \# set '--ckpt' as the correct checkpoint <br>
****

## Saliency maps
  - The saliency maps can be approached in [Baidu Cloud](https://pan.baidu.com/s/1SXAC1DtgeuyQ_WxlyI9VeQ) (fetach code is moad). Note that the results provided in paper are the average values after several training times.
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


