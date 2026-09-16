# ImageNet Classification with Deep Convolutional Neural Networks


## Introduction

<!-- Background to the study and its purpose: why hand-crafted features -->
<!-- were limiting image classification, what large labeled datasets -->
<!-- (ImageNet) made possible, and what this work set out to prove. -->
Real objects are visually much more complex than the curated ones in the MNIST dataset, which contains grayscale images of isolated digits. Recognizing real objects requires much larger training sets, but before 2012 it was not possible to collect labeled datasets with millions of images. ImageNet and LabelMe are two such datasets, and the authors trained their models on ImageNet.

The second problem is that explicitly building hand-crafted features for the 22,000 categories in ImageNet is simply impossible. It implies that the "old way" of image recognition before 2012 was engineering features one by one (possible for smaller datasets) or using other semi-automated methods. Before attempting classification, the authors had to find a class of models that fully automates feature generation. They decided to use convolutional neural networks (CNNs) for this purpose, which allow adjusting their "learning capacity" for a given task with a limited number of parameters.

This paper shows that a CNN architecture beat all competitors in classification performance on the ImageNet dataset, and goes on to describe how this was made possible. Section 3 details the network architecture and how the authors improved performance while reducing training time; Section 4 addresses solutions to overfitting. They conclude that the main bottleneck for improving CNN performance is hardware and time.

## Procedures

<!-- What the study involved: the architecture, the training setup, and -->
<!-- the tricks used to make training feasible. -->
The CNN contains eight learned layers, five convolutional and three fully-connected. The output of the last fully-connected layer is fed to a 1000-way softmax which produces a distribution over the 1000 class labels. The first convolutional layer filters the 224 × 224 × 3 input image with 96 kernels of size 11 × 11 × 3 with a stride of 4 pixels. The second convolutional layer takes as input the (response-normalized
and pooled) output of the first convolutional layer and filters it with 256 kernels of size 5 × 5 × 48.
The third, fourth, and fifth convolutional layers are connected to one another without any intervening
pooling or normalization layers. The third convolutional layer has 384 kernels of size 3 × 3 ×
256 connected to the (normalized, pooled) outputs of the second convolutional layer. The fourth
convolutional layer has 384 kernels of size 3 × 3 × 192, and the fifth convolutional layer has 256
kernels of size 3 × 3 × 192. The fully-connected layers have 4096 neurons each.

To reduce training time, the authors use Rectified Linear Units (ReLUs). This activation function converts a neuron's output as $f(x) = max(0, x)$. The authors show that it reduces the error rate of the full network several times faster than the $f(x) = |tanh(x)|$ alternative.

A single consumer-grade GPU in 2012 could hold only 1.2 million training examples in 3 GB of memory for this network. To solve this problem, they used parallelization, which allows GPUs to read from and write to one another's memory directly, without going through the host machine's slower memory.

They use a local normalization scheme called “brightness normalization,” which improves generalization, that is, it reduces the test error rate. This normalization implements a form of lateral inhibition inspired by real neurons, creating competition among strongly active neuron outputs.

Overlapping pooling summarizes the outputs of neighboring groups of neurons in the same kernel map. Using overlapping pooling instead of non-overlapping local pooling further reduces test error rates.

The authors combat overfitting in two primary ways. First, by artificially enlarging the dataset using label-preserving transformations (data augmentation). This is done with image translations and horizontal reflections, and with PCA on the set of RGB pixel values across the ImageNet training set. For each training image, they add multiples of the found principal components, with magnitudes proportional to the corresponding eigenvalues times a random variable drawn from
a Gaussian with mean zero and standard deviation 0.1. However, data augmentation is too expensive for large neural networks that already take several days to train.

Second, a technique called “dropout,” which consists of setting to zero the output of each hidden neuron with probability 0.5. The neurons which are “dropped out” in this way do not contribute to the forward pass and do not participate in back-propagation. So every time an input is presented, the neural network samples a different architecture,
but all these architectures share weights.

Finally, the authors trained their models using stochastic gradient descent with a batch size of 128 examples, momentum of 0.9, and weight decay of 0.0005.

## Results

<!-- Major findings: top-1/top-5 error rates, comparisons against prior -->
<!-- methods, which components mattered. In your own words. -->
Essentially, they beat all previous attempts at the same classification challenge by almost 20 percent for top-1 error rate and by about 35 percent for top-5 error rate. Different variants of their model still outperform the competition by about as much. Qualitatively, the CNN captures even off-center objects, and at least some errors arise when there is human-level ambiguity about the intended focus of the photograph. Further, the top-5 labels in these cases look reasonable. Crucially, at the pixel level, test and train images are visually different enough to catch the human eye, as they feature different poses, lighting, and other spatial and graphical details.

## Conclusion

<!-- The researchers' conclusions, in your own words. -->
CNNs achieve record-breaking results on large datasets of real-world objects. This specific CNN won its competition by a clear margin, and the authors say so directly. Removing any convolutional layer hurts performance; on the other hand, more training resources could have improved results even further. This is a strong result because it shows the potential of CNNs.

## Personal Notes

<!-- Your reaction: what surprised you, what stood the test of time, -->
<!-- what you would ask the authors. -->
The CNN methods described in Procedures are what we still use in 2026! I wonder what the authors would have done differently had they known the impact of their paper over the next decade. What would they change in the paper?
---
