# Installation
Our code is tested on the following environment.

## 1. Create conda environment
```bash
conda create -n quadricformer python=3.8 -y
conda activate quadricformer
```

## 2. Install PyTorch
```bash
# Choose version you want here: https://pytorch.org/get-started/previous-versions/
# We use CUDA 12.1 and PyTorch 2.1.0 for our development
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 3. Install packages from MMLab
```bash
pip install openmim
mim install mmcv==2.1.0
mim install mmsegmentation==1.2.2
mim install mmdet==3.2.0
mim install mmdet3d==1.4.0
```

## 4. Install other packages
```bash
# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu120  # choose version match your local cuda version
pip install jaxtyping timm ftfy regex einops
pip install git+https://github.com/NVIDIA/gpu_affinity
```

## 5. Install custom CUDA ops
```bash
cd model/encoder/gaussian_encoder/ops && pip install -e .
cd model/head/localagg_prob_sq && pip install -e .
```

## 6. (Optional) For visualization
```bash
pip install open3d pyvirtualdisplay matplotlib==3.7.2 PyQt5 vtk==9.0.1 mayavi==4.7.3 configobj numpy==1.23.5
```