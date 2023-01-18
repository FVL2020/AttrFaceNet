# AttrFaceNet
This is the PyTorch implementation of paper 'When Face Completion Meets Irregular Holes: An Attributes Guided Deep Inpainting Network', which can be found [here](https://dl.acm.org/doi/abs/10.1145/3474085.3475466).
# Prerequisites
* Python 3.7
* Pytorch 1.7
* NVIDIA GPU + CUDA cuDNN
# Installation
* Clone this repo:  
`git clone https://github.com/FVL2020/AttrFaceNet`
`cd AttrFaceNet-master`  
* Install Pytorch
* Install python requirements:  
`pip install -r requirements.txt`  
# Preparation
Run scripts/flist.py to generate train, test and validation set file lists. For example, to generate the training set file list on CelebA dataset, you should run:  
`python ./scripts/flist.py --path path_to_celebA_train_set --output ./datasets/celeba_train.flist`  
# Training
AttrFaceNet is trained in three stages: 1) training the attribute prediction model, 2) training the inpaint model and 3) training the joint model. To train the model:  
`python train.py --model [stage] --checkpoints [path to checkpoints]`  
# Testing
You can test the model on all three stages: 1) attribute prediction model, 2) inpaint model and 3) joint model.  
`python test.py --model [stage] --checkpoints [path to checkpoints]`
# Evaluating
Run:  
`python ./scripts/metrics.py --data-path [path to ground truth] --output-path [path to model output]`  
Then run the "read_data.m" file to obtain PSNR, SSIM and Mean Absolute Error under different mask ratios. The "log_metrics.dat" and "log_test.dat" files are in the [output-path].
To measure the Fréchet Inception Distance (FID score), run:  
`python ./scripts/fid_score.py --path [path to validation, path to model output] --gpu [GPU id to use]`  
# Citation
Please cite us if you find this work helps.  
`@inproceedings{AttrFaceNet,  
  title={When Face Completion Meets Irregular Holes: An Attributes Guided Deep Inpainting Network},  
  author={Xiao, Jie and Zhan, Dandan and Qi, Haoran and Jin, Zhi},  
  booktitle={ACM MM},  
  year={2021},  
}`  
# Appreciation
The codes refer to EdgeConnect. Thanks for the authors of it！

