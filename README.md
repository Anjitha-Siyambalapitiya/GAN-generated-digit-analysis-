# MNIST GAN and Classifier Study
## 📌 Project Overview

This project explored how a Generative Adversarial Network (GAN) can generate digit images similar to the MNIST dataset and how well a classifier trained on MNIST performs when tested on both real and fake images.

The steps followed were:

Trained a GAN on the MNIST dataset.
Used the GAN to generate fake digit images.
Trained a digit classifier on the MNIST dataset.
Tested the classifier on both real MNIST images and GAN-generated images.

## 🔬 Observations
Classifier error rate on real MNIST data: 1%
Classifier error rate on GAN-generated fake data: 4%
This shows that GAN-generated digits are fairly realistic, but they still contain small distortions that make classification slightly harder compared to real MNIST images.

## 📊 Results
GAN was successful in generating visually convincing digits.
Classifier performed slightly worse on fake images, highlighting subtle differences between GAN data and real MNIST data.
