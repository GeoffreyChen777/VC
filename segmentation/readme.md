# Virtual Category Learning: A Semi-Supervised Learning Method for Dense Prediction with Extremely Limited Labels (TPAMI under review)    

## Install

pytorch 1.13, python 3.8, cuda 11.7

```bash
python setup.py develop
```

## Prepare Data

Download data seed and cache: [link](https://1drv.ms/u/s!As5AmExWpCHXgfluc3OqaejrSYZN8w?e=kLjBiY)

```
|-- ...
|--run.py
|--data
  |-- cityscapes  
  |-- VOCSegAug
    |-- index
    |-- image
    |-- label
```


## Train
```bash
torchrun --nproc_per_node=4 run.py train --config ./config/vc_deeplabv3p/voc_aug.yaml --num-gpus=4
```
