# [[CVPR 2026]]() ZeroIDIR: Zero-Reference Illumination Degradation Image Restoration with Perturbed Consistency Diffusion Models
<h4 align="center">Hai Jiang<sup>1</sup>, Zhen Liu<sup>2</sup>, Yinjie Lei<sup>3</sup>, Songchen Han<sup>1</sup>, Bing Zeng<sup>2</sup>, Shuaicheng Liu<sup>2</sup></center>
<h4 align="center">1.School of Aeronautics and Astronautics, Sichuan University</center></center>
<h4 align="center">2.University of Electronic Science and Technology of China,</center></center>
<h4 align="center">3.College of Electronics and Information Engineering, Sichuan University</center></center>

## Overall pipeline
![](./Figure/pipeline.png)

## Dependencies
```
pip install -r requirements.txt
````

## Download the raw training and evaluation datasets
### LLIE datasets
#### ● Paired datasets 
● LOL dataset: Chen Wei, Wenjing Wang, Wenhan Yang, and Jiaying Liu. "Deep Retinex Decomposition for Low-Light Enhancement". BMVC, 2018. [[Baiduyun (extracted code: sdd0)]](https://pan.baidu.com/s/1spt0kYU3OqsQSND-be4UaA) [[Google Drive]](https://drive.google.com/file/d/18bs_mAREhLipaM2qvhxs7u7ff2VSHet2/view?usp=sharing)

● LSRW dataset: Jiang Hai, Zhu Xuan, Ren Yang, Yutong Hao, Fengzhu Zou, Fang Lin, and Songchen Han. "R2RNet: Low-light Image Enhancement via Real-low to Real-normal Network". Journal of Visual Communication and Image Representation, 2023. [[Baiduyun (extracted code: wmrr)]](https://pan.baidu.com/s/1XHWQAS0ZNrnCyZ-bq7MKvA)

### BIE datasets

### MSEC datasets

### Real-world datasets

## Pre-trained Models 
You can download our pre-trained model from [[OneDrive]]() and [[Baidu Yun (extracted code:)]]()

## How to train?
You need to modify ```dataset/dataloader.py``` slightly for your environment, and then
```
python train.py  
```

## How to test?
```
python inference.py
```

## Visual comparison
![](./Figure/visual_compare.png)
## Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```

```

## Acknowledgement
Part of the code is adapted from the previous work: [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch). We thank all the authors for their contributions.

