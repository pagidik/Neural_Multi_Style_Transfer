# Neural_Multi_Style_Transfer
Transferring style from multiple images
<<<<<<< HEAD
This repository is based on the original paper Neural Style Transfer
original NST paper, CVPR, new : https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
Citation : Leon A. Gatys, Alexander S. Ecker, Matthias Bethge

# Intoduction to NST
The algorithm transfers style from one input image (the style image) onto another input image (the content image) using CNN nets (usually VGG-16/19) and gives a composite, stylized image out which keeps the content from the content image but takes the style from the style image.
=======

This repository contains the PyTorch implementation of the original Neural Style Transfer Paper.
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf

# Running the script
  clone the repository to your local folder. 
  install the dependencies as per the enviroment variables. 
  run 'python3 neural_style_transfer.py' command to execute the script. 

# How is this different from original paper?
The original paper transfers style form one input image (the style image) onto another input image ( the content image). In this work, we have modified the algorithm to take input from two style images and transfer it onto the content image. 

The original paper is is written in TensorFlow, we have implemented in PyTorch and also used L-BFGS optimizer. This optimizer is much faster than the adam optimizer. 
The loss function is weighted on each of the content loss and two style losses. This way we can control the amount of style transfer from each input style image.

The issue with this implementaion is that it takes alot of computation power if we increase the size of the image.  So we had to resize the image to make it smaller. 
Due to this resize the resolution of the image is compromised. So we wanted to bring the original image by passing the output of this network into a SR model. 

We have re-implemented the SRGAN paper on our own.
https://arxiv.org/abs/1609.04802

However we could not make the the SR GAN paper to execute successfully. 

>>>>>>> 513a376 (Readme)
