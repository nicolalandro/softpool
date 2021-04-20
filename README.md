# Softpool
This code is a fork of [softpool official code](https://github.com/wangyida/softpool) of the [ECCV 2020 paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123480069.pdf).

## Requirements
```
#python3.6
python3.6 -m venv venv
source venv/bin/activate
pip install -r requirements

cd distance/emd
python setup.py install
cd ../chamfer
python setup.py install
# maybe you need to put some env
# PATH=/usr/local/cuda-10.0/bin:$PATH LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH CUDA_HOME="$CUDA_HOME/usr/local/cuda-10.0" python setup.py install
cd ../../

cd MSN/expansion_penalty
python setup.py install
cd ../../

cd MSN/MDS
python setup.py install
cd ../../
```

## Train
```
# enable venv
CUDA_VISIBLE_DEVICES=0 python3 train.py --batch 16 --n_regions 8 --num_points 2048 --dataset shapenet --savepath ijcv_shapenet_softpool --methods softpool
```
