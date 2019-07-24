# Depth Completion with Error-Map
Tensorflow implementation of our paper [Conf-Net: Predicting Depth Completion Error-Map For
High-Confidence Dense 3D Point-Cloud](https://arxiv.org).

# Introduction
This work proposes a new method for depth completion of sparse LiDAR data using a convolutional neural network which learns to generate ”almost” full 3D point-clouds with significantly lower root mean squared error (RMSE) over state-of-the-art methods. Our main contributions are listed below:

 Markup : * We propose a novel method to predict a high-quality pixel-wise error-map. Our approach outperforms existing methods in terms of uncertainty and confidence maps.
          * Our approach generates industry-level clean (high confidence - low variance) 360 ◦ 3D dense point-cloud from sparse LiDAR point-cloud. Our point-cloud is 15 times denser than input (which is Velodyne HDL 64 point-cloud) and 3 times more accurate than the state-of-the-art (RMSE = 300mm).
          * We conduct the uncertainty based analysis of Kitti depth completion dataset for the first time.


See full demo on Youtube.
<p align="center">
  <img src="images/artak.gif">
</p>


*Sparse Depth:*

<p align="center">
  <img src="images/raw.gif">
</p>

*Predicted Dnese Depth:*

<img align="cener" src="images/mean.gif">

*Predicted Pixelwise Error-Map:*

<img align="cener" src="images/var.gif">

<img align="cener" src="images/table.png">

<!--*Point-Cloud in 3D:*

<img width="420" align="cener" src="images/demo2.gif">
<img width="420" align="cener" src="images/demo3.gif">
<img width="420" align="cener" src="images/demo1.gif">
<img width="420" align="cener" src="images/artak.gif">
-->
## Installing

Just run the docker.
```
bash docker run
```
## Training

```
python main.py train
```
## Testing
```
python deploy_haval.py
```

