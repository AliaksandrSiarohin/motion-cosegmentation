# Motion Supervised co-part Segmentation

This repository contains the source code for the paper [Motion Supervised co-part Segmentation]() by Aliaksandr Siarohin<sup>\*</sup>, Subhankar Roy<sup>\*</sup>, [Stéphane Lathuilière](https://stelat.eu/), [Sergey Tulyakov](http://stulyakov.com), [Elisa Ricci](http://elisaricci.eu/) and [Nicu Sebe](http://disi.unitn.it/~sebe/).

<sup>\*</sup> - denotes equal contribution

Our method is a self-supervised deep learning method for co-part segmentation. Differently from previous works, our approach develops the idea that motion information inferred from videos can be leveraged to discover meaningful object parts. Our method can also perform video editing (aka part-swaps).

## Example segmentations

Unsupervised segmentations obtained with our method on <b>VoxCeleb</b>:

<table>
  <tr>
    <td><image src="sup-mat/vox-seg-0.gif" width="256" /></td>
    <td><image src="sup-mat/vox-seg-1.gif" width="256"/></td>
    <td><image src="sup-mat/vox-seg-2.gif" width="256"/></td>
    <td><image src="sup-mat/vox-seg-3.gif" width="256"/></td>
    <td><image src="sup-mat/vox-seg-4.gif" width="256"/></td>
    <td><image src="sup-mat/vox-seg-5.gif" width="256"/></td>
  </tr>
</table>

and <b>TaiChi</b> dataset:
<table>
  <tr>
    <td><image src="sup-mat/taichi-seg-0.gif" width="256"/></td>
    <td><image src="sup-mat/taichi-seg-1.gif" width="256"/></td>
    <td><image src="sup-mat/taichi-seg-2.gif" width="256"/></td>
    <td><image src="sup-mat/taichi-seg-3.gif" width="256"/></td>
    <td><image src="sup-mat/taichi-seg-4.gif" width="256"/></td>
    <td><image src="sup-mat/taichi-seg-5.gif" width="256"/></td>
  </tr>
</table>

## Example part-swaps

Part swaping with our method for VoxCeleb dataset. Each triplet shows source image, target video (with swap mask in the corner) and result:

<table style="width:100%;;padding-top:35%">
  <tr>
    <th colspan="3"> Hair Swap </th>
    <th colspan="3"> Beard Swap </th>
  <tr/>


  <tr>
    <td colspan="3"><image src="sup-mat/hair-line.gif" width="768"/></td>
    <td colspan="3"><image src="sup-mat/beard-line.gif" width="768"/></td>
  </tr>

  <tr>
    <th colspan="3"> Eyes Swap </th>
    <th colspan="3"> Lips Swap </th>
  <tr/>

  <tr>
    <td colspan="3"><image src="sup-mat/eye-line.gif" width="768"/></td>
    <td colspan="3"><image src="sup-mat/lips-line.gif" width="768"/></td>
  </tr>


</table>

### Installation

We support ```python3```. To install the dependencies run:
```
pip install -r requirements.txt
```

### YAML configs

There are several configuration (```config/dataset_name.yaml```) files one for each `dataset`. See ```config/taichi-sem-256.yaml``` to get description of each parameter.

### Pre-trained checkpoints
Checkpoints can be found under following links: [yandex-disk](https://yadi.sk/d/2hTyhEcqo_5ruA) and [google-drive](https://drive.google.com/open?id=1SsBifjoM_qO0iFzb8wLlsz_4qW2j8dZe).

### Part-swap demo

To run a demo, download checkpoint and run the following command:
```
python part_swap.py  --config config/dataset_name.yaml --target_video path/to/target --source_image path/to/source --checkpoint path/to/checkpoint --swap_index 0,1
```
The result will be stored in ```result.mp4```.

* For swaping either soft or hard labels can be used (specify ```--hard``` for hard segmentation).

* For swaping either target or source segmentation mask can be used (specify ```--use_source_segmentation``` for using source segmentation mask).

* For the reference we also provide fully-supervised segmentation. For fully-supervised add --supervised option. And run
```git clone https://github.com/AliaksandrSiarohin/face-makeup.PyTorch face_parsing```
which is a fork of @zllrunning. 

* Also for the reference we provide [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) based alignment, use ```--first-order-motion-model``` and the correspoinding checkpoint. This allignment can only be used along with ```--suprevised``` option.


### Colab Demo

We prepare a special demo for the google-colab, see: ```part_swap.ipynb```.

### Training
Model training consist in finetuning the First Order Model checkpoint (they can be downloaded from [google-drive](https://drive.google.com/open?id=1PyQJmkdCsAkOYwUyaj_l-l0as-iLDgeH) or [yandex-disk](https://yadi.sk/d/lEw8uRm140L_eQ)). Use the following command for training:
```
CUDA_VISIBLE_DEVICES=0 python train.py --config config/dataset_name.yaml --device_ids 0 --checkpoint dataset-name.cpk.pth.tar
```
The code will create a folder in the log directory (each run will create a time-stamped new directory).
Checkpoints will be saved to this folder. To check the loss values during training in see ```log.txt```.
You can also check training data reconstructions in the ```train-vis``` subfolder.
By default the batch size is tunned to run on 1 Tesla-p100 gpu, you can change it in the train_params in the corresponding ```.yaml``` file.


### Datasets

1) **Taichi**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.

2) **VoxCeleb**. Please follow the instruction from https://github.com/AliaksandrSiarohin/video-preprocessing.


### Training on your own dataset
1) Follow instructions from [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) for preparing your dataset and train [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) on your dataset.

2) This repository use the same dataset format as [First Order Motion Model](https://github.com/AliaksandrSiarohin/first-order-model) so you can use the same data as in 1).

#### Additional notes

Citation:

Motion Supervised co-part Segmentation:
```
@article{Siarohin_2020_motion,
  title={Motion Supervised co-part Segmentation},
  author={Siarohin, Aliaksandr and Roy, Subhankar and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  journal={arXiv preprint},
  year={2020}
}

```
First Order Motion Model:
```
@InProceedings{Siarohin_2019_NeurIPS,
  author={Siarohin, Aliaksandr and Lathuilière, Stéphane and Tulyakov, Sergey and Ricci, Elisa and Sebe, Nicu},
  title={First Order Motion Model for Image Animation},
  booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
  month = {December},
  year = {2019}
}

```
