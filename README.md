# 253 Project

### Xin Lin, Yufan Wei, Haomin Qi


## Abstract
Deep learning, particularly convolutional neural networks (CNNs), achieves superior image denoising compared with traditional model-based approaches. However, many CNN denoisers obtain accuracy gains by increasing network depth, which raises computational cost and can hinder generalization. The Multi-stage Wavelet-based Dynamic Convolutional Neural Network (MWDCNN) addresses this trade-off by integrating dynamic convolution with wavelet transforms to balance performance and efficiency. In this work, we extend MWDCNN from a single-task denoising network to a multi-task image restoration framework, MWRCNN (Multi-stage Wavelet-based Restoration CNN). MWRCNN generalizes the original architecture to handle deblurring, JPEG artifact removal, and super-resolution while retaining the multi-stage structure and wavelet-based frequency analysis. We focus on the Dynamic Convolutional Block (DCB) and its Weight Generator (WG), which produces input-adaptive convolution kernels. We analyze limitations of the current lightweight WG design—namely its tendency to oversimplify spatial information—and propose strategies that preserve richer spatial context without sacrificing computational efficiency. Our goal is to improve both adaptability and overall restoration quality of MWRCNN across diverse low-level vision tasks.

## Dataset:


## Pre-trained model:

