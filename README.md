# NBNet: Noise Basis Learning for Image Denoising with Subspace Projection 

Code for CVPR21 paper [NBNet](https://arxiv.org/abs/2012.15028).

## Training

### Preparation

```
python prepare_data.py --data_dir yours_sidd_data_path
```



### Begin training:

For SIDD benchmark,  use:

```
python train_mge.py -d prepared_data_path -n num_gpus
```



For DnD benchmark,  we use MixUp additionally:

```
python train_mge.py -d prepared_data_path -n num_gpus --dnd
```



## Pretrained model

MegEngine checkpoint for SIDD benchmark  can be found [here](https://drive.google.com/file/d/1RPAf9ZJqqq9ePPVTtJRlixX4-h3HJTCc/view?usp=sharing)
