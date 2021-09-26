# MSRF-Net: A Multi-Scale Residual Fusion Network for Biomedical Image Segmentation
This repository provides code for our paper "MSRF-Net: A Multi-Scale Residual Fusion Network for Biomedical Image Segmentation" ([arxiv version](https://arxiv.org/abs/2105.07451))  
## 2.) Overview#
### 2.1.)Introduction

In   this   work,   we   propose   a   novel   medical   imagesegmentation  architecture,  calledMSRF-Net,  which  aims  toovercome  the  above  limitations.  Our  proposed  MSRF-Netmaintains  high-resolution  representation  throughout  the  pro-cess  which  is  conducive  to  potentially  achieving  high  spatialaccuracy.  The  MSRF-Net  utilizes  a  novel  dual-scale dense fusion (DSDF) block that performs dual scale feature exchangeand  a  sub-network  that  exchanges  multi-scale  features  usingthe  DSDF  block.  The  DSDF  block  takes  two  different  scaleinputs and employs a residual dense block that exchanges in-formation across different scales after each convolutional layerin  their  corresponding  dense  blocks.  The  densely  connectednature  of  blocks  allows  relevant  high-  and  low-level  featuresto be preserved for the final segmentation map prediction. Themulti-scale  information  exchange  in  our  network  preservesboth high- and low-resolution feature representations, therebyproducing  finer,  richer,  and  spatially  accurate  segmentationmaps. The repeated multi-scale fusion helps in enhancing thehigh-resolution  feature  representations  with  the  informationpropagated  by  low-resolution  representations.  Further,  layersof residual networks allow redundant DSDF blocks to die out,and only the most relevant extracted features contribute to thepredicted segmentation maps.
## 2.2.) DSDF Blocks and MSRF Sub-network
![](Fig2_new-page-001.jpg)
## 2.3.) Quantitative Results


## Data Preparation
1.) make directory named "data/isic"

2.) make three sub-directories "train" "valid" "test"

3.) Put images under directory named "image"

4.) Put images under directory named "mask"


Run the script as:
python msrf.py
