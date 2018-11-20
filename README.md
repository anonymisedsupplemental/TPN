## Transductive Propagation Network
Code for "Learning to Propagate Labels: Transductive Propagation Network for Few-shot Learning"

## Requirements
Python 3.5
Tensorflow 1.3+
tqdm


## Data (miniImagenet and tieredImagenet)
Please download the compressed tar files from: https://github.com/renmengye/few-shot-ssl-public

mkdir -p data/miniImagenet/data

tar -zxvf mini-imagenet.tar.gz

mv *.pkl data/miniImagenet/data

mkdir -p data/tieredImagenet/data

tar -xvf tiered-imagenet.tar

mv *.pkl data/tieredImagenet/data

### TPN mini-5way1shot
```
python train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w1s_5tw1ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```

```
python test.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w1s_5tw1ts_rn300_k20 --rn=300 --alpha=0.99 --k=20 --iters=81500
```

### TPN mini-5way5shot
```
python train.py --gpu=0 --n_way=5 --n_shot=5 --n_test_way=5 --n_test_shot=5 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w5s_5tw5ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```

```
python test.py --gpu=0 --n_way=5 --n_shot=5 --n_test_way=5 --n_test_shot=5 --lr=0.001 --step_size=10000 --dataset=mini --exp_name=mini_TPN_5w5s_5tw5ts_rn300_k20 --rn=300 --alpha=0.99 --k=20 --iters=50100

```

### TPN tiered-5way1shot
```
python train.py --gpu=0 --n_way=5 --n_shot=1 --n_test_way=5 --n_test_shot=1 --lr=0.001 --step_size=25000 --dataset=tiered --exp_name=tiered_TPN_5w1s_5tw1ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```

### TPN tiered-5way5shot
```
python train.py --gpu=0 --n_way=5 --n_shot=5 --n_test_way=5 --n_test_shot=5 --lr=0.001 --step_size=25000 --dataset=tiered --exp_name=tiered_TPN_5w5s_5tw5ts_rn300_k20 --rn=300 --alpha=0.99 --k=20
```
