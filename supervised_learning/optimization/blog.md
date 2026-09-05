# Optimization Techniques: From Data Prep to Learning Rate Decay

To start, all the techniques we discuss concern different steps of the training process: first comes feature scaling, which is a preprocessing step before training. Second, we have batch normalization, which operates during the forward pass by normalizing each layer's inputs. The rest concern the optimization step after backpropagation: mini-batch gradient descent decides how much data each gradient is computed on, momentum, RMSProp and Adam decide how gradients are turned into updates, and learning rate decay decides how the learning rate evolves over time. All of them aim to find the optimal weights (W) and biases (b) for the neural network model.

**Feature scaling** concerns the range of the input features (not the labels or outcome variable). It is thus a data preprocessing technique. In most cases, we achieve scaling by standardization (X - mean)/sigma, which gives mean 0 and variance 1, or by min-max normalization (X - min)/(max - min), which typically squeezes the variable between 0 and 1 or any specified range. The reason is that gradient descent minimizes the loss better when all input features are on the same scale, otherwise the widest-range feature dominates and descent zig-zags.

Pro: It rounds out the cost bowl so gradient descent takes bigger, straighter steps and converges faster.
Con: You must reuse the training mean and variance on test data, and min-max scaling is fragile to outliers.

**Batch normalization** normalizes the activations at every layer of the network, so each layer trains on inputs with a stable distribution instead of chasing the shifting outputs of earlier layers. Because the inputs stay well-behaved, it allows larger learning rates and makes very deep networks trainable. Any regularization from the per-batch noise is only a small side effect, not its purpose, so it is not a replacement for Lasso, Ridge, or dropout.

Pro: It gives every layer stable inputs, so training is faster with larger learning rates, and at test time it folds into the weights so there is no extra runtime cost.
Con: It depends on batch statistics, so tiny batches make the estimates noisy and test time needs stored running averages instead of a real batch.

**Batch, mini-batch gradient descent, and epochs.** Batch gradient descent (BGD) computes the gradient over the entire training set before each update, which gives a stable step but is slow and can get stuck in poor minima on big data. Stochastic gradient descent goes to the other extreme with one example per update, which is fast to start but too noisy. **Mini-batch gradient descent** is the compromise: it computes the gradient and updates the parameters on a small, randomly selected subset called a mini-batch, so it makes many vectorized updates per epoch, where one epoch is a single full pass over the training set.

Pro: It makes many updates per epoch, escapes poor minima more easily than batch GD, and the data no longer needs to fit in memory, with sizes like 64 to 512 powers of 2 as a good default.
Con: The cost curve is noisy from batch to batch, so a single step going up does not mean the learning rate is wrong.

**Gradient descent with momentum** keeps a running average of past gradients, so recent steps count a lot and older steps fade away, like inertia in physics or an AR(1) smoother in time series. The algorithm remembers the direction of previous updates, so it builds up speed in consistent directions and cancels out oscillations in noisy directions such as narrow ravines.

Pro: It denoises the noisy mini-batch gradients and powers through ravines and plateaus instead of bouncing wall to wall.
Con: It can overshoot the minimum and adds a momentum coefficient beta to tune, usually around 0.9.

**RMSProp optimization** gives each parameter its own effective learning rate by dividing its gradient by the root of a moving average of its recent squared gradients, resulting in a process akin to down-weighting noisy observations. Directions with large oscillating gradients get divided down, while quiet directions keep relatively larger steps, so one global learning rate works across uneven terrain.

Pro: It equalizes progress per weight, so steep directions stop oscillating and flat directions keep moving.
Con: It still needs a global learning rate to be tuned, and the forgetting factor decides how fast old gradients are forgotten.

**Adam optimization** combines momentum and RMSProp with bias correction: it keeps a moving average of the gradients and a moving average of the squared gradients, corrects both for their zero start, and updates each weight by the corrected average divided by the root of the corrected squared average. That makes it an automatic default that crosses plateaus and saddle points with little tuning beyond the learning rate.

Pro: It auto-combines momentum and per-weight scaling with bias correction, so it works well out of the box across architectures.
Con: It can generalize slightly worse than SGD with momentum on some tasks, which is why fixes like decoupled weight decay (AdamW) or switching to SGD late exist.

**Learning rate decay** slowly reduces the learning rate over training: big steps early to cover ground, small steps late to settle into the minimum instead of wandering around it from mini-batch noise. A well-chosen fixed learning rate matters first, and decay is the final polish on top of it.

Pro: It anneals the step size from large to small, so training starts fast and finishes settled tightly in the minimum.
Con: It adds another schedule to tune, so fix the base learning rate first before reaching for decay.
