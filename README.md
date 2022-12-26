# CDCN
This is a pytorch implementation of [Bridging Component Learning with Degradation Modelling for Blind Image Super-Resolution](https://ieeexplore.ieee.org/abstract/document/9925720).
This repo is built on the basis of [DAN](https://github.com/greatlog/DAN) and [BasicSR](https://github.com/XPixelGroup/BasicSR), thanks for their open-sourcing!
## Requirement
+ python3
+ NVIDIA GPU + CUDA
+ pytorch >= 1.7.1
+ python packages: ``` pip install -r requirements.txt ```
+ bascisr: ``` python setup.py develop ```
## Train
Download the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and [Flickr2K](http://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) and merge it into one folder. Modify `options/train/train_setting.yml` and run the following command
```
python basicsr/train.py -opt=options/train/train_setting.yml
```
## Test
There are two blind settings mentioned in our paper. For setting1, we synthesize the *Gaussian8* datasets with five datasets: Set5, Set14, BSD100, Urban100, Manga109. Please refer to [this repository](https://github.com/Arcananana/DSSR) for more details.
For setting2, we using the benchmark dataset [DIV2KRK]((http://www.wisdom.weizmann.ac.il/~vision/kernelgan/DIV2KRK_public.zip)) from [KernelGAN](https://github.com/sefibk/KernelGAN).
The pretrained models can be downloaded [here](https://pan.baidu.com/s/1K2Qi4ejzQPnLC7m5_8UlsQ?pwd=cdcn) (setting1 x2 model is missed and we are re-training it).
Modify the dataset path and test settings in `options/test/test_setting.yml` and run the following command
```
python basicsr/test.py -opt=options/test_setting.yml
```
## Citation
If you find this repo useful, please consider citing our work:
```
@ARTICLE{9925720,
  author={Wu, Yixuan and Li, Feng and Bai, Huihui and Lin, Weisi and Cong, Runmin and Zhao, Yao},
  journal={IEEE Transactions on Multimedia}, 
  title={Bridging Component Learning with Degradation Modelling for Blind Image Super-Resolution}, 
  year={2022},
  volume={},
  number={},
  pages={1-16},
  doi={10.1109/TMM.2022.3216115}}
```
