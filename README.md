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
mv ~/downloads/checkpoint0029_4scale_swin.pth ~/Setting-Detection-and-Tracker/DINO/weights/
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






