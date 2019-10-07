# Micro-Net-DNW

In this submission to the Micro-Net challenge we train a sparse version of ResNet50 v1.5.
The method for sparsifying the ResNet 50 v1.5 appears in our paper on
[Discovering Neural Wirings](https://arxiv.org/abs/1906.00586) to appear at NeurIPS 2019.

We train a ResNet50 v1.5 which has only 10% of the total weights remaining. As standard, the first layer, batchnorm layers, and biases are left dense.

This repo, however, is based on a standard repo by NVIDIA for training ResNet50 v1.5, which may be found [here](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/RN50v1.5).
We begin by providing some preliminary information from their README.

# 1. Perliminary Information from the NVIDIA repository.
## Requirements

Ensure you meet the following requirements:

* [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)
* [PyTorch 19.06-py3 NGC container](https://ngc.nvidia.com/registry/nvidia-pytorch) or newer

## Data Augmentation

This model uses the following standard data augmentation:

* For training:
  * Normalization
  * Random resized crop to 224x224
    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3
  * Random horizontal flip

* For inference:
  * Normalization
  * Scale to 256x256
  * Center crop to 224x224

# 2. Training

To train a model you need 4 GPUs. 

First, `cd` into this directory and enter the docker container via the following code.

```
nvidia-docker run -it --rm --privileged --ipc=host \
    -v `pwd`:/workspace/rn50 \
    -v <path to imagenet on your computer>:/data/imagenet \
    nvcr.io/nvidia/pytorch:19.06-py3

cd rn50
```

You may now train your model, which should take about 2 days.

```bash
bash exp/starter.sh ignorefirst10 <folder-where-you-want-to-save-checkpoints>
```

# 3. Testing & Pretrained Model

The model we trained can be downloaded [here](https://drive.google.com/file/d/1PIX1BJX72BIM-t6kHtarK1vdvM4C2T0j/view?usp=sharing).
Put the model in this directory so it is visible in the docker container.

First, `cd` into this directory and enter the docker container via the following code.

```
nvidia-docker run -it --rm --privileged --ipc=host \
    -v `pwd`:/workspace/rn50 \
    -v <path to imagenet on your computer>:/data/imagenet \
    nvcr.io/nvidia/pytorch:19.06-py3

cd rn50
```
Test with the following command, which requires only one GPU:
```
bash exp/test.sh ignorefirst10 <path-to-model>
```
The script should output the following as one of the last lines, which contains the top-1 accuracy.

```
Summary Epoch: 0/90;	val.top1 : 75.228	val.top5 : 92.620	val.loss : 1.058
```

Additionally, if you look at the output of the evaluation script, you will see the number of params, FLOPS, and speed documented.
This is visible right after the words `begining to train`.
See `image_classification/model_profiling.py` for the standard profiler we use.

These are
```
 params: 2,558,732
 flops: 515,231,310
 nanoseconds: 77,699,608 
```

# 4. MicroNet Challenge Score

We have computer flops as multiply-adds. We may therefore double it to get an upper bound on multiply + adds.
Accordingly, our math operations is 1,030,462,620. Our parameters is 2,558,732.

Therefore, our score is 
```
(2,558,732 / 6,900,000) + (1,030,462,620/ 1,170,000,000) = 1.251568
```