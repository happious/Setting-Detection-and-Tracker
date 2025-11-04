## 1.Installation
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 \
  --extra-index-url https://download.pytorch.org/whl/cu116
```

```git clone https://github.com/happious/Setting-Detection-and-Tracker```

```pip install numpy cython
pip install -r requirements.txt```

```cd DINO/models/dino/ops
python setup.py build install
python test.py      # "All checking is True" 나오면 성공
cd ../../..```

## 2.DENO
```python realtime.py```



