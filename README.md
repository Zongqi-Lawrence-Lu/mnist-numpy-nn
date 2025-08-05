# Handwritten Digit Recognition with NumPy

This project implements a deep fully-connected neural networkfrom scratch in NumPy to classify digits from the MNIST dataset (http://yann.lecun.com/exdb/mnist/).

The goal is to understand core concepts of deep learning by building every component manually — no high-level ML frameworks used.


# Development Timeline

Version | Features 
  v1.0  | One-layer shallow network
  v2.0  | Generalized to arbitrary-depth networks; He initialization 
  v2.1  | Added momentum, stochastic gradient descent, and softmax output 
  v2.2  | Enabled mini-batch training for faster training


# How to Run

1. Download the MNIST dataset 
2. Read it (handled automatically by "Load_MNIST.py").
3. Run "train.py" to train the model.


# Results

- Training time for 50 epochs: < 3 minutes on a single core CPU. 
- The model consistently fits the training data to 100% accuracy.
- Test accuracy remains robust at ~97% using a 2-hidden-layer network (Sample training log in "sample_output.")


# Current Status

The project is complete and performs well for the MNIST benchmark.  
No further improvements are planned — a fully connected network is sufficient for a project like this.


# Motivation

This was my first ML project after a hands-on AI/ML in Biology workshop.  
Built while reading Chapter 3-7 of Understanding Deep Learning by Simon Prince.
