# [CVPR 2026] ZeroIDIR: Zero-Reference Illumination Degradation Image Restoration with Perturbed Consistency Diffusion Models [[Paper]]()
<h4 align="center">Hai Jiang<sup>1</sup>, Zhen Liu<sup>2</sup>, Yinjie Lei<sup>3</sup>, Songchen Han<sup>1</sup>, Bing Zeng<sup>2</sup>, Shuaicheng Liu<sup>2</sup></center>
<h4 align="center">1.School of Aeronautics and Astronautics, Sichuan University</center></center>
<h4 align="center">2.University of Electronic Science and Technology of China,</center></center>
<h4 align="center">3.College of Electronics and Information Engineering, Sichuan University</center></center>

## Dataset synthesis pipeline
![](./Figure/syn_pipe.png)

## Framework pipeline
![](./Figure/framework.png)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### LLIE datasets
Our SIED dataset is available at [[OneDrive]](https://1drv.ms/f/c/e379fe7c770e3033/Ejd2sO7WNMlGiAVSqFtu1KkBF8UL_RU9unCds1Mu8z8IPw?e=gyBKoy) and [[Baidu Yun (extracted code:4y4w)]](https://pan.baidu.com/s/13DpBAePEHpV0k4Mj96fgrw). Please see the txt files in ```data``` folder for the training set and evaluation set split. 

### BIE datasets

### MSEC datasets

### Real-world datasets

## Pre-trained Models 
You can download our pre-trained model from [[OneDrive]]() and [[Baidu Yun (extracted code:m9zp)]]()

## How to train?
You need to modify ```dataset/dataloader.py``` slightly for your environment, and then
```
accelerate launch train.py  
```

## How to test?
```
python inference.py
```

## Visual comparison
![](./Figure/visual_canon.png)
![](./Figure/visual_sony.png)
## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```

```

## Acknowledgement
Part of the code is adapted from the previous work: [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). We thank all the authors for their contributions.

