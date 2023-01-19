# AttrFaceNet
This is the PyTorch implementation of paper 'When Face Completion Meets Irregular Holes: An Attributes Guided Deep Inpainting Network', which can be found [here](https://dl.acm.org/doi/abs/10.1145/3474085.3475466).
### Introduction:
We propose a novel facial attributes guided face completion network named AttrFaceNet. The proposed AttrFaceNet is composed of two subnets. One is for facial attribute prediction while the other one is for face completion. The attribute prediction subnet learns to output 38 predicted facial attributes (e.g., male, smiling, eyeglasses, etc.) from the corrupted image. For better prediction, these 38 attributes are firstly divided into nine groups according to their characteristics or locations on the face and then are predicted by nine branches respectively. Moreover, due to intrinsic relations between these attributes, e.g.,
“Male” is inherently related to “No Beard”, a fully connected layer is added at the end of this subnet to collaboratively fuse the nine group attributes. The face completion subnet is a U-Net architecture, which firstly downsamples the feature maps and then upsamples them to the original size. During the upsampling process, the former predicted attributes are also upsampled through deconvolutional layers and then injected as guidance into the face completion subnet at every upsampling scale.
![framework](https://raw.githubusercontent.com/FVL2020/AttrFaceNet/main/figs/framework.png)  
# Prerequisites
* Python 3.7
* Pytorch 1.7
* NVIDIA GPU + CUDA cuDNN
# Installation
* Clone this repo:  
```
git clone https://github.com/FVL2020/AttrFaceNet
cd AttrFaceNet-master
```
* Install Pytorch
* Install python requirements:
```
pip install -r requirements.txt
```
# Preparation
Run scripts/flist.py to generate train, test and validation set file lists. For example, to generate the training set file list on CelebA dataset, you should run:  
```
python ./scripts/flist.py --path path_to_celebA_train_set --output ./datasets/celeba_train.flist
```
# Training
AttrFaceNet is trained in three stages: 1) training the attribute prediction model, 2) training the inpaint model and 3) training the joint model. To train the model:
```
python train.py --model [stage] --checkpoints [path to checkpoints]
```
# Testing
You can test the model on all three stages: 1) attribute prediction model, 2) inpaint model and 3) joint model.
```
python test.py --model [stage] --checkpoints [path to checkpoints]
```
# Evaluating
Run:
```
python ./scripts/metrics.py --data-path [path to ground truth] --output-path [path to model output]
```
Then run the "read_data.m" file to obtain PSNR, SSIM and Mean Absolute Error under different mask ratios. The "log_metrics.dat" and "log_test.dat" files are in the [output-path].   
To measure the Fréchet Inception Distance (FID score), run:
```
python ./scripts/fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]
```
# Results
### Quantitative Results:
![quantitative_results](https://raw.githubusercontent.com/FVL2020/AttrFaceNet/main/figs/quantitative_results.png)  
Quantitative comparisons of our AttrFaceNet with state-of-the-art methods on CelebA and Helen datasets. Noting that the FID results on Helen dataset are not provided since the number of images is not sufficient enough to compute an accurate FID. The best and second best results are marked in red and blue, respectively.
### Qualitative Results:
![qualitative_results](https://raw.githubusercontent.com/FVL2020/AttrFaceNet/main/figs/qualitative_results.png)  
Qualitative comparisons of our AttrFaceNet with state-of-the-art methods.
# Citation
Please cite us if you find this work helps.  
```
@inproceedings{AttrFaceNet,
  title={When Face Completion Meets Irregular Holes: An Attributes Guided Deep Inpainting Network},
  author={Xiao, Jie and Zhan, Dandan and Qi, Haoran and Jin, Zhi},
  booktitle={ACM MM},
  year={2021},
}
```
# Appreciation
The codes refer to EdgeConnect. Thanks for the authors of it！
# License
This repository is released under the MIT License as found in the LICENSE file. Code in this repo is for non-commercial use only.
