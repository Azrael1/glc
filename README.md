# Gold Loss Correction - Tensorflow

This repository contains Tensorflow code for the paper

[Using Trusted Data to Train Deep Networks on Labels Corrupted by Severe Noise (NeurIPS 2018)](http://arxiv.org/abs/1802.05300).

The code requires Python 3+, PyTorch [0.3, 0.4), and TensorFlow 1.14.

Pytorch & Official Version of code is available [here](https://github.com/mmazeika/glc).

The objective of the code is to run a minimal ConvNet model on Tensorflow, and compare the results before & after implementing GLC. It has not replicated the results of the paper because the paper uses a bigger convnet(Wideresnet), as opposed to the one that I use. The model still achieves a lift from using GLC. 

The method used to implement is 'glc', as opposed to using the 'confusion matrix' directly from the data.
