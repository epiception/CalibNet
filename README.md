# CalibNet

### [DEPRECATED] This repository is no longer actively supported. 

While the authors work on an update, please check out this unofficial implementation: [CalibNet_pytorch](https://github.com/gitouni/CalibNet_pytorch) :fire: :slightly_smiling_face:
___

Code for our paper:
[CalibNet: Self-Supervised Extrinsic Calibration using 3D Spatial Transformer Networks](https://arxiv.org/pdf/1803.08181.pdf)

Check out our [project page](https://epiception.github.io/CalibNet/)!

![CalibNet_gif1](https://media.giphy.com/media/1zjOgLf7j4lHmeMubG/giphy.gif)

### Prerequisites
CalibNet is trained on Tensorflow 1.3, CUDA 8.0, CUDNN 7.0.1


##### Installation

The code for point cloud distance loss is modified from [PU-NET](https://github.com/yulequan/PU-Net), PointNet++, PointSetGeneration.

This repository, thus, is based on Tensorflow and the TF operators from PointNet++ and PU-NET.

For installing tensorflow, please follow the official instructions in here. The code is tested under TF1.3 and Python 2.7 on Ubuntu 16.04.

For compiling TF operators, please check tf_xxx_compile.sh under each op subfolder in code/tf_ops folder, and change the path correctly to ../path/to/tensorflow/include. Note that you need to update nvcc, python and tensoflow include library if necessary. You also need to remove -D_GLIBCXX_USE_CXX11_ABI=0 flag in g++ command in order to compile correctly if necessary.

We are working to update the code and installation steps for the latest tensorflow versions.

### Dataset Preparation

To prepare the dataset, run /dataset_files/dataset_builder_parallel.sh in the directory where you wish to store. We will also create a parser `parsed_set.txt` for the dataset, that contains the file names for training.

```
git clone https://github.com/epiception/CalibNet.git
or
svn checkout https://github.com/CalibNet/trunk/code (for the code)
cd ../path/to/dataset_directory
bash ../path/to/code_folder/dataset_files/dataset_builder_parallel.sh
cd ../path/to/CalibNet/code
python dataset_files/parser.py ../dataset_directory/2011_09_26/
```
##### Resnet-18
Pretrained Resnet-18 parameters can be found [here](https://drive.google.com/open?id=1XGqdBH3A88m1LgUIe5tS7VjKjtQc1A6V).


### Training

Before training, be sure to make requisite changes to the paths and training parameters in the config file `config_res.py`.
We trained using 2 GPUs. The base code is written to support ops for the same device configuration. 

To begin training:
```
CUDA_VISIBLE_DEVICES=<device_id1>,<device_id2> python -B train_model_combined.py
```

##### Trained Weights
Trained weights for the base variant (non-iterative) model is available [here](https://drive.google.com/drive/folders/138hq7OgTEBmG-wK52h7gchg5ob1WqARn?usp=sharing). This model was trained for 44 epochs. As mentioned in the paper, the iterative realignment model for better translation outputs will uploaded soon.

### Evaluation/Test
Code for Direct Evaluation/Testing pipeline for point cloud calibration will be uploaded soon.
