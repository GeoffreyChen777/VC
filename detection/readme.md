# Virtual Category Learning

This is the code for the virtual category learning.

## Detection

### Prepare Data

Download data and cache: [OneDrive](https://1drv.ms/u/s!As5AmExWpCHXgdI2u7uNIZaFkKBGIA?e=pmFmXY)

```
|-- ...
|--run.py
|--data
  |--COCO2017
|--cache
  |-- coco17_x_x.dict
  |-- ...
  |-- resnet50.pth
```


### Train
```bash
python run.py train --num_gpus 8 --config ./config/virtual_category/coco.yaml
```

### Test

```bash
python run.py test --num_gpus 8 --config ./config/virtual_category/coco.yaml --resume ./log/vc/01/final.pth
```

### Citation

```bibtex
@inproceedings{chen2022ssodvc,
    year = 2022,
    title = {Semi-supervised Object Detection via Virtual Category Learning.},
    author = {Changrui Chen and Kurt Debattista and Jungong Han},
    booktitle = {European Conference on Computer Vision (ECCV)}
}
```