# Convolutional Neural Networks — Resources

Project: Convolutional Neural Networks (intranet 2303) · Dir: `supervised_learning/cnn`
Ingested: 2026-09-13 · Session stayed alive for the full run (health check passed, all 20 rltokens resolved in-context).
Sections on project page: Read or watch (15 items incl. 4-video sublist) · Definitions to skim (none — section absent) · References (9 labels, 5 unique URLs).

## Convolutional neural network

URL: https://en.wikipedia.org/wiki/Convolutional_neural_network  ·  Date: 2026-09-03 (last edited)  ·  Status: summary

A CNN is a feedforward network that learns features by optimizing convolution filters, and it remains the reference architecture for vision tasks even as transformers take over some niches. The typical stack is an input layer, hidden layers mixing convolution, pooling, fully connected and normalization layers, and an output layer. Two ideas carry the design: each neuron sees only a local receptive field, and all neurons in a feature map share one filter, which collapses the parameter count and stabilizes training.

- Shared weights slash parameters: a 5x5 filter needs 25 weights where a fully connected neuron on a 100x100 image would need 10,000.
- Core hyperparameters: padding, stride, filter count, kernel size, pooling type/size, dilation.
- Pooling is max or average; it downsamples and cuts compute but strict translation invariance is lost to the downsampling itself.
- Roots run from visual-cortex receptive fields through Fukushima's models to LeNet-5, then the GPU era (AlexNet).
- Task link: conv/pool forward and backward passes (tasks 0-3) and the Keras LeNet-5 build (task 4).

## The best explanation of Convolutional Neural Networks on the Internet!

URL: https://medium.com/technologymadeeasy/the-best-explanation-of-convolutional-neural-networks-on-the-internet-fbb8b1ad5df8  ·  Date: 2016-07-28 (publication)  ·  Status: summary

A beginner-oriented explainer (intranet itself jokes it is good but not literally the best) built around one idea: CNNs operate over volumes, not flat vectors. A filter slides across the multi-channel image, each dot product yields one scalar, and the full sweep forms a feature map; independent filters produce independent maps. Early layers settle into edge and color-blob detectors while deeper layers assemble those fragments into larger parts. Parameter sharing plus local connectivity keep the model small, max pooling trims spatial size, and a standard head of repeated conv/activation/pool blocks feeding fully connected layers completes the classifier.

- Volume input (e.g. height x width x 3) is the break from plain MLPs.
- One filter sweep = one feature map; six filters = six maps.
- First-layer filters train into edges and blobs; depth builds composite parts.
- Zero padding is deliberately skipped here; read about it separately.
- Builds on the Stanford cs231n CNN notes (linked as its reference).

## Machine Learning is Fun! Part 3: Deep Learning and Convolutional Neural Networks

URL: https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721  ·  Date: 2016-06-13 (publication)  ·  Status: summary

A gentle tutorial that earns the convolution by showing what fails first: a plain pixel-fed net recognizes a centered digit yet breaks when the digit shifts. Brute fixes such as sliding-window search or piling on data and depth (GPUs included) are deemed insufficient. The proposed pipeline tiles the image with overlap, runs the same small network on every tile with shared weights, reassembles the scores into a grid, max-pools it down, and classifies with a fully connected net; stacking such stages yields deep feature hierarchies. A worked bird classifier on CIFAR-10 plus Caltech-UCSD Birds with TFLearn/TensorFlow lands near 95.5 percent, and the closing lesson is that accuracy alone misleads.

- Synthetic training data (generated shifts/scales) stretches small datasets.
- Convolution buys translation invariance: an object stays itself wherever it sits.
- Max pooling here means keeping the largest value of each 2x2 square.
- Depth ladder: edges, then beaks from edges, then whole birds.
- Judge classifiers with true/false positives/negatives, precision and recall, not headline accuracy.

## Convolutional Neural Networks: The Biologically-Inspired Model

URL: https://www.codementor.io/@james_aka_yale/convolutional-neural-networks-the-biologically-inspired-model-iq6s48zms  ·  Date: 2018-04-19 (publication; page notes update 2018-10-15)  ·  Status: summary

A long-form survey that opens with human scene understanding (a Harry Potter still) to motivate bio-inspired design, then defines CNNs through many identical copies of one neuron with shared parameters. It recaps the tile-convolve-reassemble-pool-classify pipeline, traces LeCun's Bell Labs check reader and the 1998 end-to-end paper with Bottou, Haffner and Bengio, and dissects the four blocks: convolution, ReLU, pooling, fully connected layers. The second half contrasts AlexNet with LeNet and tours applications across vision, language and speech.

- ReLU displaced LeNet's sigmoid, whose saturation, slow convergence and non-zero-centered outputs hurt training.
- Architecture splits into feature extraction (conv plus pooling) and classification (fully connected layers).
- AlexNet upgrades: deeper, more and stacked filters, ReLU, dropout, ImageNet-scale data.
- Vision lineage afterward: ZFNet, GoogLeNet, VGGNet, ResNet, DenseNet.
- Applied reach: R-CNN detection family, tracking, segmentation (FCN, SegNet), captioning (LRCN), fast neural translation, sentence classification, QA, and CNN-HMM speech systems.

## Back Propagation in Convolutional Neural Networks — Intuition and Code

URL: https://becominghuman.ai/back-propagation-in-convolutional-neural-networks-intuition-and-code-714ef1c38199  ·  Date: 2017-12-14 (publication)  ·  Status: summary

A compact derivation of conv-layer backprop from the chain rule, worked on a minimal single-channel case: 3x3 input, one 2x2 filter, stride 1, no padding, producing a 2x2 output. The forward pass caches inputs and filters; the backward pass assumes the upstream gradient arrives and solves for filter and input gradients, leaning on the insight that each filter weight touches every output pixel so all those contributions accumulate. The piece flags that its operation is cross-correlation (no filter flip), points its code at the deeplearning.ai CNN course assignment, and closes with links to deeper derivations and a lecture on convnet backprop. Retrieval note: plain fetch returned a stub and archive.org was blocked, so the full text (complete, 800 words) was recovered inside the authenticated browser session.

- Cache X and W during the forward pass for reuse in the backward pass.
- One weight affects all output pixels; its gradient sums every path.
- Output gradient doubles as the previous layer's input gradient — the engine of backprop.
- Cross-correlation naming kept explicit (unflipped filter).
- Task link: the conv_backward derivation for task 2.

## Backpropagation in a convolutional layer

URL: https://medium.com/data-science/backpropagation-in-a-convolutional-layer-24c8d64d8509  ·  Date: 2019-07-10 (publication)  ·  Status: summary

A rigorous, notation-first derivation using the cs231n convention: inputs (N, C, H, W), filters (F, C, HH, WW), biases (F,), stride and symmetric padding, with output size 1 + (H + 2*pad − HH) / stride. It climbs from a 1-D vector case to a 2-D matrix case before generalizing to depth, landing three rules: bias gradients sum the upstream gradient over space, filter gradients convolve the input with the upstream gradient as filter, and input gradients convolve padded upstream gradient with the flipped filter. A naive implementation is validated against numerical gradients with errors near 1e-9, scoped to stride 1.

- Adopt the (N, C, H, W) / (F, C, HH, WW) bookkeeping before touching indices.
- db: one sum of dy per filter map; dw: input-star-dy pattern; dx: padded-dy against rotated filter.
- Einstein summation keeps the matrix-case algebra readable.
- Gradient checking is the acceptance test for any hand implementation.
- Task link: shape discipline for conv_backward (task 2).

## Convolutional Neural Network (CNN) — Backward Propagation of the Pooling Layers

URL: https://lanstonchu.wordpress.com/2018/09/01/convolutional-neural-network-cnn-backward-propagation-of-the-pooling-layers/  ·  Date: 2018-09-01 (publication)  ·  Status: summary

A focused companion to conv backprop that visualizes only the pooling backward pass, motivated by the observation that most tutorials leave pooling gradients to frameworks. Average pooling divides the upstream gradient evenly, adding dA/(H*W) to every cell of each window; max pooling routes the whole gradient to the argmax cell via a boolean mask and zeros elsewhere. Both accumulate into dA_prev with overlap-aware addition, illustrated with NumPy snippets on rank-4 tensors. It notes LeNet-5's average pooling heritage against max pooling's modern dominance.

- Forward average: window mean; backward: spread dA/(H*W) over the window.
- Forward max: window argmax; backward: mask times dA at the winning cell only.
- Always accumulate (+=) since strided windows overlap.
- Rank-4 (m, h, w, c) indexing is the main notational burden, not the calculus.
- Task link: the pooling backward formulas for task 3.

## Backpropagation In Convolutional Neural Networks (DeepGrid — Pooling Layer section)

URL: https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/#pooling-layer  ·  Date: 2016-09-05 (publication)  ·  Status: summary

A full mathematical treatment of conv forward and backward passes; the intranet's "Pooling Layer" label lands on this article's pooling section. It separates convolution from cross-correlation by the flipped kernel, derives filter gradients as a flipped-delta convolution (the double sum coming straight from weight sharing) and input gradients from a flipped-kernel cross-correlation, then states pooling rules crisply: no learning happens there, max pooling routes error to the cached winning unit, average pooling spreads error scaled by 1/n^2. Sparse connectivity plus shared weights plus pooling are credited with translation invariance.

- Convolution equals cross-correlation with a horizontally and vertically flipped kernel.
- Weight sharing is why filter gradients sum contributions from all output positions.
- Max backprop: error to the recorded winner; average backprop: uniform split.
- Sparse local connectivity is what makes deep stacks affordable.
- Companion reference: the Dumoulin/Visin convolution-arithmetic guide.

## Gradient-Based Learning Applied to Document Recognition (LeNet-5 paper)

URL: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf  ·  Date: 1998-11 (paper publication, Proc. IEEE 86(11))  ·  Status: summary

The landmark LeCun, Bottou, Bengio and Haffner paper behind the intranet's LeNet-5 label (summary grounded in the verified IEEE abstract; the hosted PDF uses custom font encodings that defeat text extraction). It reviews handwritten-character methods on the standard digit task and shows CNNs, built for 2-D shape variability, beating every alternative. It then introduces graph transformer networks for globally training multi-module recognition systems (extraction, segmentation, language modeling) and reports deployed systems including a bank-cheque reader processing millions of cheques daily.

- Venue: Proceedings of the IEEE, vol. 86, no. 11, pp. 2278-2324, November 1998.
- Claim: CNNs outperform all other techniques on the digit benchmark.
- GTNs extend gradient learning to whole multi-module pipelines.
- Proof of deployment: commercial cheque-reading at massive scale.
- Task link: the architecture task 4 reimplements in Keras.

## deeplearning.ai (course platform hub)

URL: https://www.deeplearning.ai/  ·  Date: unknown  ·  Status: summary

The parent link wrapping the four Ng videos below resolves to the DeepLearning.AI homepage: course catalog, specializations, short courses and the Batch newsletter rather than lesson content itself. Its value here is provenance — the four videos are week-1 material of the CNN course of the Deep Learning Specialization — plus the intranet's practical tip to watch them at 1.5x-2x speed.

- Hub page, not a lesson; substance lives in the four sublisted videos.
- Identifies the specialization context (C4W1 lecture codes).
- No byline or modified date available; marked unknown rather than guessed.

## Convolutional Neural Networks (CNNs) explained (deeplizard)

URL: https://www.youtube.com/watch?v=YRhxdVk_sIs  ·  Date: 2017-12-09 (YouTube upload)  ·  Status: summary

A visual primer: CNNs are ANNs specialized in pattern detection, dominant in image analysis but applicable elsewhere, with convolutional layers as their defining part. Each layer declares a number of filters — small randomly initialized matrices — that convolve across the input computing dot products, whose grid of results becomes the next layer's input. A worked MNIST-style "7" demo plus hand-set edge filters shows early layers catching edges, corners and circles while deeper layers compose eyes, fur, dog faces and bird legs.

- Filter = small matrix; convolving = sliding plus dot products.
- Hierarchy: geometric primitives first, object parts later, whole objects deepest.
- Brightest output pixels mark what a filter detected.
- Credits a Jeremy Howard (fast.ai) lecture for the walkthrough device.
- Runtime about 8.5 minutes; points onward to coding videos.

## Why Convolutions (DeepLearningAI, C4W1L11)

URL: https://www.youtube.com/watch?v=ay3zYUeuyhU  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

Ng's case for convolutions over fully connected layers rests on two mechanisms. Parameter sharing reuses one feature detector at every image position on the grounds that an edge or face cue useful in one corner is probably useful in another; sparse connections tie each output only to its local patch. The worked contrast is stark: wiring a 32x32x3 input to a 28x28x6 output densely costs on the order of 14 million weights, while the convolutional version needs only hundreds. The payoff is trainability on smaller datasets with less overfitting, plus robustness to small shifts, and training itself stays familiar: choose the conv/pool/fully-connected layout, initialize, define cost over the labeled set, optimize with gradient descent or momentum/RMSprop/Adam variants.

- Sharing: one detector, all positions, both low- and high-level features.
- Sparsity: each output sees only its receptive patch, ignoring the rest.
- Fewer parameters permit smaller training sets and curb overfitting.
- Shared filters at every position bake in shift robustness.
- Same training loop as classic nets: cost J plus a gradient optimizer.

## One Layer of a Convolutional Net (DeepLearningAI, C4W1L07)

URL: https://www.youtube.com/watch?v=jPOAS7uCODQ  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

A line-by-line construction of a single conv layer: convolve a 6x6x3 volume with two 3x3x3 filters to get two 4x4 maps, broadcast-add one scalar bias per filter, apply ReLU, stack into a 4x4x2 activation. Ng maps this onto the classical z = Wx + b, a = g(z) step — convolution plays the linear role, bias and nonlinearity keep theirs. A parameter exercise (ten 3x3x3 filters = 280 parameters) shows the count stays fixed even for gigantic images, and the lecture closes by fixing notation: filter size F[l], padding p[l] (valid vs same), stride s[l], per-layer volume dims, the floor((n + 2p − F)/s + 1) sizing formula, channel-matching filters, and a warning that channel-first vs channel-last ordering varies across codebases.

- Bias is one real number broadcast over its whole map.
- Output channels always equal the filter count.
- Filter depth must equal the incoming channel count.
- 280 parameters regardless of image size is the anti-overfitting argument in miniature.
- Task link: the (m, h, w, c) layout and same/valid padding of task 0.

## Simple Convolutional Network Example (DeepLearningAI, C4W1L08)

URL: https://www.youtube.com/watch?v=3PyJA9AfwSk  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

A first complete network for binary image classification on a 39x39x3 input: a same-convolution with ten 3x3 filters (stride 1) to 37x37x10, a 5x5 stride-2 conv with 20 filters to 17x17x20, another 5x5 stride-2 conv with 40 filters to 7x7x40, then flattening (1960 units) into a logistic or softmax head. The design moral is the standard depth trend — spatial dims shrink while channels grow — and the layer vocabulary is set to three types: convolutional, pooling, fully connected, with hyperparameter-choice guidance deferred to later videos.

- Sizing formula applied end to end: (39 + 0 − 3)/1 + 1 = 37 and so on.
- Flattening 7x7x40 gives the 1960-vector feeding the classifier.
- Trend: height/width fall (39 → 37 → 17 → 7), channels rise (3 → 10 → 20 → 40).
- Most architecture work is hyperparameter selection, not novel blocks.
- Task link: the flatten-then-dense tail reused in the Keras LeNet-5 (task 4).

## CNN Example (DeepLearningAI, C4W1L10 — LeNet-5 style)

URL: https://www.youtube.com/watch?v=bXJx7y51cl0  ·  Date: 2017-11-07 (YouTube upload)  ·  Status: summary

A LeNet-5-inspired digit classifier on 32x32x3 input: 5x5 conv with 6 filters (valid) to 28x28x6 with ReLU, 2x2 stride-2 max pooling to 14x14x6, 5x5 conv with 16 filters to 10x10x16, pooling to 5x5x16, flatten to 400, fully connected 120 then 84 (weight matrix 120x400 at the first dense step), softmax over 10 digits. Ng also settles the layer-counting question by counting only weight-bearing layers (pooling folds into its conv layer) and observes the usual profiles: pooling adds no parameters, conv layers add few, dense layers hold most, activations taper downward.

- Conv/pool pairs halve spatial dims while preserving channels.
- 5x5x16 flattens to exactly the 400-vector entering FC3.
- Count layers with weights; conventions differ across papers and posts.
- Prefer borrowing a published architecture over inventing hyperparameters.
- Task link: the closest conceptual dry run of the task-4 LeNet-5 build.

## References (link-only, all verified resolving)

- tf.layers.Conv2D / tf.keras.layers.Conv2D → https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D (HTTP 200 · page last modified 2024-06-07). Note: both intranet labels land on the Keras Conv2D page.
- tf.layers.AveragePooling2D / tf.keras.layers.AveragePooling2D → https://www.tensorflow.org/api_docs/python/tf/keras/layers/AveragePooling2D (HTTP 200 · page last modified 2024-06-07).
- tf.layers.MaxPooling2D / tf.keras.layers.MaxPooling2D → https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D (HTTP 200 · page last modified 2024-06-07). Note: target class is named MaxPool2D (no "ing") — match the API spelling in code.
- tf.layers.Flatten / tf.keras.layers.Flatten → https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten (HTTP 200 · page last modified 2024-06-07).
- Reproducibility in Keras Models → https://keras.io/examples/keras_recipes/reproducibility_recipes/ (HTTP 200 · page last modified 2026-08-26). Needed for deterministic training runs.

## Quiz Hooks

- Parameter sharing — one filter reused at every position slashes weights and fights overfitting.
- Local connectivity / receptive field — each neuron sees only a small patch of the previous layer.
- Feature map — the grid of dot products from sweeping one filter over the input.
- Valid vs same convolution — no padding versus padding that preserves spatial size.
- Output sizing — floor((n + 2p − F)/s + 1) governs each spatial dimension.
- Max vs average pooling — keep the window maximum or the window mean.
- Pooling backprop — route gradient to the argmax cell (max) or spread dA/(H*W) (average).
- Conv backprop trio — bias sums dy, filter gradient is input-star-dy, input gradient uses the flipped filter.
- Translation invariance — a shifted object keeps its identity and, ideally, its label.
- ReLU over sigmoid — avoids saturation, trains faster, zero-centers better in deep stacks.
- Flatten — unroll the final volume (e.g. 5x5x16 = 400) into the dense head.
- LeNet-5 — 1998 conv/pool pioneer for document and digit recognition, the task-4 blueprint.
- Precision and recall — the honest report card beyond headline accuracy.
