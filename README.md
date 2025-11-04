## 0.File
<https://drive.google.com/drive/folders/1qD5m1NmK0kjE5hh-G17XUX751WsEG-h_>

Download **checkpoint0029_4scale_swin.pth**


## 1.Installation
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

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

```
cd models/dino/ops
python setup.py build install
python test.py
cd ../../..
```

## 2.DENO
```
python realtime.py
```



## Jetson
버전 확인
```
sudo apt list nvidia-jetpack
```

### 1.JetPack 5.1 또는 5.1.2 (CUDA 11.4 / 11.8, Python 3.8)
```
pip3 install torch==1.13.0+nv23.05 torchvision==0.14.0+nv23.05 \
  --extra-index-url https://pypi.ngc.nvidia.com
```

### 2. JetPack 6.0 (CUDA 12.2, Python 3.10)
```
pip3 install torch==2.1.0+nv24.05 torchvision==0.16.0+nv24.05 \
  --extra-index-url https://pypi.ngc.nvidia.com
```

CUDA 인식 확인
```
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```


### 3. Requirements

ONNX
```
pip3 install onnxruntime-gpu-aarch64==1.14.1
```

초기설정
```
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev g++ cmake libopenblas-base libopenmpi-dev
pip3 install --upgrade pip setuptools wheel
```

```
numpy
cython
git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI&egg=pycocotools
submitit
git+https://github.com/cocodataset/panopticapi.git#egg=panopticapi
scipy
termcolor
addict
yapf
timm
opencv_python
loguru
scikit-image
tqdm
Pillow
thop
ninja
tabulate
tensorboard
lap
filterpy
h5py
```







