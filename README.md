# 0.File
<https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_>

Download **checkpoint0029_4scale_swin.pth**

<br><br><br>

# 1.Installation

## 1. RTX 30 Series
**test : Python=3.8, PyTorch=1.12.1, Torchvision=0.13.1, CUDA=11.6**


### i. Setting
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

### ii. Build package
```
git clone https://github.com/happious/Setting-Detection-and-Tracker
```

```
cd Setting-Detection-and-Tracker
pip install numpy cython
pip install -r requirements.txt
```

```
cd DINO
mkdir weights
mv ~/Downloads/checkpoint0029_4scale_swin.pth ~/Setting-Detection-and-Tracker/DINO/weights/
```

### iii. Install DINO
```
cd models/dino/ops
python setup.py build install
python test.py
cd ../../../..
```




  
## 2. RTX 40 Series
**test : Python=3.10, PyTorch=2.4.0, Torchvision=0.19.0, CUDA=12.2**


### i. python 3.10
```
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
libnss3-dev libssl-dev libreadline-dev libffi-dev libbz2-dev libsqlite3-dev \
wget curl liblzma-dev tk-dev
```

```
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
sudo tar -xf Python-3.10.14.tgz
cd Python-3.10.14
```

```
sudo ./configure --enable-optimizations --with-ensurepip=install
sudo make -j$(nproc)
sudo make altinstall
```

```
python3.10 -m ensurepip
python3.10 -m pip install --upgrade pip setuptools wheel
```

```
python3.10 --version
```

### ii. PyTorch+CUDA
```
python3.10 -m pip install torch==2.4.0+cu122 torchvision==0.19.0+cu122 --index-url https://download.pytorch.org/whl/cu122

```

```
# Check
python3.10 - <<'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
EOF
```

```
# Ex
Torch: 2.4.0+cu122
CUDA runtime: 12.2
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

### iii. Build package
```
git clone https://github.com/happious/Setting-Detection-and-Tracker
```

```
cd Setting-Detection-and-Tracker
python3.10 -m pip install \
numpy \
cython \
"git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools" \
submitit \
"git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi" \
scipy \
termcolor \
addict \
yapf \
timm \
opencv-python \
loguru \
scikit-image \
tqdm \
Pillow \
thop \
ninja \
tabulate \
tensorboard \
lapx \
filterpy \
h5py
```

```
cd DINO
mkdir weights
mv ~/Downloads/checkpoint0029_4scale_swin.pth ~/Setting-Detection-and-Tracker/DINO/weights/
```

### iv. Install DINO
```
cd models/dino/ops
python3.10 setup.py build install --user
python3.10 test.py
cd ../../../..
```



  
# 2.DEMO
```
python realtime.py
```













