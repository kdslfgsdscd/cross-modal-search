# Introduction

## Abstract:
> 本发明的一种基于子空间适应性间距的跨模态检索方法及存储介质，提出基于有监督子空间适应性间距的跨模态检索方法，用于解决现有跨模态哈希检索方法中存在的检索精度低的技术问题，为项目实施过程中的跨模态信息检索提供关键算法。发明主要包括以下步骤，数据预处理，进行训练集测试集数据划分，并提取数据的原始高维特征；输入原始高维特征到网络模型获取图像文本的公共特征和对应的预测标签信息；使用公共特征和标签信息计算每种模态不同类别样本的适应性间距损失，然后结合注意力机制聚焦图片和文本中类别信息用于增强不同类别的判别性，最后计算不同模态之间的不变性损失；再通过反向传播对损失函数进行优化去迭代网络模型至收敛；使用收敛的网络模型计算所有图像文本的公共特征；最后对查询数据特征与公共特征进行相似度计算并排序返回结果。采用本发明进行跨模态检索的精度高于现有方法进行跨模态检索的精度。

## Conda env:
```
name: dscmr
channels:
  - pytorch
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - blas=1.0=mkl
  - brotlipy=0.7.0=py37h7b6447c_1000
  - ca-certificates=2020.6.24=0
  - certifi=2020.6.20=py37_0
  - cffi=1.14.1=py37he30daa8_0
  - chardet=3.0.4=py37_1003
  - cryptography=2.9.2=py37h1ba5d50_0
  - cudatoolkit=10.1.243=h6bb024c_0
  - freetype=2.10.2=h5ab3b9f_0
  - idna=2.10=py_0
  - intel-openmp=2020.1=217
  - jpeg=9b=h024ee3a_2
  - lcms2=2.11=h396b838_0
  - ld_impl_linux-64=2.33.1=h53a641e_7
  - libedit=3.1.20191231=h14c3975_1
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.1.0=hdf63c60_0
  - libpng=1.6.37=hbc83047_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - libtiff=4.1.0=h2733197_1
  - lz4-c=1.9.2=he6710b0_1
  - mkl=2020.1=217
  - mkl-service=2.3.0=py37he904b0f_0
  - mkl_fft=1.1.0=py37h23d657b_0
  - mkl_random=1.1.1=py37h0573a6f_0
  - ncurses=6.2=he6710b0_1
  - ninja=1.10.0=py37hfd86e86_0
  - olefile=0.46=py37_0
  - openssl=1.1.1g=h7b6447c_0
  - pillow=7.2.0=py37hb39fc2d_0
  - pip=20.2.1=py37_0
  - pycparser=2.20=py_2
  - pyopenssl=19.1.0=py_1
  - pysocks=1.7.1=py37_1
  - python=3.7.7=hcff3b4d_5
  - pytorch=1.6.0=py3.7_cuda10.1.243_cudnn7.6.3_0
  - readline=8.0=h7b6447c_0
  - requests=2.24.0=py_0
  - setuptools=49.2.1=py37_0
  - six=1.15.0=py_0
  - sqlite=3.32.3=h62c20be_0
  - tk=8.6.10=hbc83047_0
  - torchvision=0.7.0=py37_cu101
  - tqdm=4.48.2=py_0
  - urllib3=1.25.10=py_0
  - wheel=0.34.2=py37_0
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
  - zstd=1.4.5=h9ceee32_0
  - pip:
    - absl-py==0.10.0
    - astunparse==1.6.3
    - boto==2.49.0
    - boto3==1.14.47
    - botocore==1.17.47
    - cachetools==4.1.1
    - cssselect==1.1.0
    - cycler==0.10.0
    - docutils==0.15.2
    - gast==0.3.3
    - gensim==3.8.3
    - google-auth==1.20.1
    - google-auth-oauthlib==0.4.1
    - google-pasta==0.2.0
    - grpcio==1.31.0
    - h5py==2.10.0
    - imageio==2.9.0
    - importlib-metadata==1.7.0
    - jmespath==0.10.0
    - joblib==0.16.0
    - keras==2.4.3
    - keras-preprocessing==1.1.2
    - kiwisolver==1.2.0
    - lxml==4.7.1
    - markdown==3.2.2
    - matplotlib==3.3.0
    - numpy==1.18.5
    - nvidia-dali-cuda100==1.3.0
    - nvidia-ml-py3==7.352.0
    - oauthlib==3.1.0
    - opencv-python==4.5.4.60
    - opt-einsum==3.3.0
    - pandas==1.1.0
    - protobuf==3.13.0
    - pyasn1==0.4.8
    - pyasn1-modules==0.2.8
    - pyparsing==2.4.7
    - pyquery==1.4.3
    - python-dateutil==2.8.1
    - pytz==2020.1
    - pyyaml==5.3.1
    - requests-oauthlib==1.3.0
    - rsa==4.6
    - s3transfer==0.3.3
    - scikit-learn==0.23.2
    - scipy==1.4.1
    - sentencepiece==0.1.91
    - sklearn==0.0
    - smart-open==2.1.0
    - tensorboard==2.3.0
    - tensorboard-plugin-wit==1.7.0
    - tensorflow==2.3.0
    - tensorflow-estimator==2.3.0
    - termcolor==1.1.0
    - threadpoolctl==2.1.0
    - torchtext==0.2.3
    - werkzeug==1.0.1
    - wrapt==1.12.1
    - zipp==3.1.0
prefix: /home/anaconda3/envs/dscmr
```

## Dataset url:
***Wikipedia***

[link](http://www.svcl.ucsd.edu/projects/crossmodal/)

***pascal***

[link](http://vision.cs.uiuc.edu/pascal-sentences/)


## Running:

> Training
>
> python main.py
>
> Testing
> python evaluate.py

