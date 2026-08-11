# Classification Using Neural Networks — Resources

## Read or watch

### Forward propagation
URL: https://www.youtube.com/watch?v=wL17g67vU88  ·  Date: 2018-01-05 (YouTube upload)  ·  Status: summary
AskDevOps walks through the forward pass of a small network: the weighted sum z = w·x + b at each neuron, then the activation applied on top. The video stresses the layer-by-layer repetition of this same two-step computation, from the input features through the hidden layers to the final output.
- forward pass = repeat "weighted sum, then activate" at every layer
- z is computed first, then a = sigmoid(z); both are needed for backprop
- the same pattern scales to networks of any depth

### Backpropagation calculus
URL: https://www.youtube.com/watch?v=tIeHLnjs5U8  ·  Date: 2017-11-03 (YouTube upload)  ·  Status: summary
3Blue1Brown derives backpropagation as nothing but the chain rule applied to a chain of simple operations. Each weight's contribution to the cost is computed by multiplying local derivatives along the path from that weight to the output, and the video shows exactly why the activation used determines the derivative you multiply by.
- backprop = repeated chain rule, no new math beyond derivatives
- the "error" at each neuron is the aggregate derivative flowing into it
- sigmoid's derivative being σ(z)(1−σ(z)) makes the algebra collapse to a multiply

### What is a Neural Network?
URL: https://www.youtube.com/watch?v=n1l-9lIMW7E  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Andrew Ng's motivation video: neural networks as functions that map inputs to outputs, trained with enough labeled data (x, y) pairs. A housing-price example shows how hidden units can learn concepts like "family size" or "walkability" and combine them into a final prediction.
- a network learns the mapping f(x) → y from data alone
- hidden units learn intermediate concepts; output combines them
- more data and larger networks both improve the learned mapping

### Supervised Learning with a Neural Network
URL: https://www.youtube.com/watch?v=BYGpKPY9pO0  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Intro to supervised learning as the dominant real-world use of neural networks: the model is trained on labeled examples and predicts for new inputs. Examples span housing-price prediction (regression), advertising click-through, image tagging, and speech recognition.
- supervised learning needs labeled input-output pairs
- regression predicts continuous values; classification predicts categories
- different network structures suit different data types (structured, image, audio)

### Binary Classification
URL: https://www.youtube.com/watch?v=eqEc66RFY0I  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Ng frames binary classification with the cat-vs-no-cat image problem: input x is a flattened 64×64×3 pixel vector of length n_x = 12288, and the target y is 0 or 1. This sets up the shapes used throughout the course — column vectors for x, and the m-example matrix X of shape (n_x, m).
- one training example: x is a column vector (n_x, 1); y ∈ {0, 1}
- X = [x¹ x² … xᵐ] stacks examples as columns, shape (n_x, m)
- Y holds all labels as a row, shape (1, m)

### Logistic Regression
URL: https://www.youtube.com/watch?v=hjrYrynGWGA  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Logistic regression as a tiny learning algorithm: parameters w (same shape as x) and scalar b, with prediction ŷ = σ(wᵀx + b) squeezed into (0, 1) so it can be read as a probability. The logistic/sigmoid function is what makes the linear score a probability.
- ŷ = σ(wᵀx + b), with σ(z) = 1/(1 + e^(−z))
- w is a vector of the same length as x; b is a scalar bias
- output is interpreted as P(y = 1 | x), the probability of the positive class

### Logistic Regression Cost Function
URL: https://www.youtube.com/watch?v=SHEPb1JHw5o  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Why squared error is a bad loss for logistic regression (non-convex, many local minima) and how the log loss fixes it. The loss per example is −(y log ŷ + (1−y) log(1−ŷ)); the cost J averages the loss over all m training examples, which keeps the objective convex.
- loss measures one example; cost J = (1/m) Σ loss over the whole set
- log loss punishes confident wrong answers heavily
- convexity of J is why gradient descent reliably finds the minimum

### Gradient Descent (Ng)
URL: https://www.youtube.com/watch?v=uJryes5Vk1o  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
The training loop for finding the parameters that minimize J(w, b): start anywhere (the cost is convex, so any start converges), repeatedly step in the direction of steepest downhill slope. The update w := w − α·dw, b := b − α·db uses the learning rate α to control step size and the derivative to set direction.
- update rule: parameter minus learning rate times the derivative
- dw/db are the code names for the derivative terms
- α too small → slow; too large → overshoot and diverge

### Computation Graph
URL: https://www.youtube.com/watch?v=hCP1vGoCdYU  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Ng introduces computational graphs to organize a forward pass (compute J) and a backward pass (compute derivatives). A forward propagation step computes outputs; a backward propagation step walks the graph in reverse, applying the chain rule to get every dJ/dW and dJ/db.
- forward pass computes the cost; backward pass computes derivatives
- backprop is the chain rule walking the graph in reverse order
- one graph organises both passes — no extra machinery needed

### Logistic Regression Gradient Descent
URL: https://www.youtube.com/watch?v=z_xiwjEdAC4  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Full derivative workout for the single training example: from dz = a − y (the elegant simplification of the log-loss gradient) through dw = x·dz and db = dz, then the gradient-descent updates. This is the pattern backprop generalizes: the derivative of the loss with respect to the output is the seed of every gradient.
- dz = ŷ − y: gradient of the loss w.r.t. z
- dw = x·dz and db = dz per example
- two updates: w := w − α·dw, b := b − α·db

### Vectorization
URL: https://www.youtube.com/watch?v=qsIrQi0fzbY  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Vectorization is the art of removing explicit for loops so numpy can use SIMD-style parallel instructions on CPU and GPU. Ng's demo shows a dot product of million-element arrays: ~1.5 ms vectorized vs ~400-500 ms with a Python loop, roughly 300× slower. Rule of thumb: whenever possible, avoid explicit for loops.
- numpy built-ins run thousands of times faster than Python loops
- both CPU and GPU benefit; GPUs are just better at SIMD
- same values either way — vectorizing changes speed, not results

### Vectorizing Logistic Regression
URL: https://www.youtube.com/watch?v=okpqeEUdEkY  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Ng vectorizes the forward pass over all m examples: Z = WᵀX + b becomes one matrix operation, with numpy broadcasting adding the scalar b to every column, then A = σ(Z) applying the sigmoid element-wise. One line replaces an m-iteration loop.
- Z = WᵀX + b: one matmul covers all m examples (X is (n_x, m))
- broadcasting adds b across columns automatically
- A = σ(Z) applies the activation element-wise

### Vectorizing Logistic Regression's Gradient Computation
URL: https://www.youtube.com/watch?v=2BkqApHKwn0  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
The backward pass, vectorized: dZ = A − Y is a single (1, m) matrix operation, dw = (1/m)X·dZᵀ averages over examples, and db = (1/m)·np.sum(dZ). Two lines replace the loop over all examples.
- dZ = A − Y across all examples at once
- dw = (1/m) X dZᵀ; db = (1/m) Σ dZ
- gradient descent then: w −= α·dw, b −= α·db

### A Note on Python/Numpy Vectors
URL: https://www.youtube.com/watch?v=V2QlTmh6P2Y  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Ng warns against rank-1 arrays like a = np.random.randn(5) — neither row nor column, whose shape is (5,). Using a.reshape(5, 1) or (1, 5) keeps broadcasting predictable and avoids bugs. assert(a.shape == (5,1)) is a cheap sanity check.
- rank-1 arrays cause silent broadcasting bugs; always reshape
- prefer explicit column vectors (n, 1) and row vectors (1, n)
- assert on shapes during debugging

### Neural Network Representations
URL: https://www.youtube.com/watch?v=CcRkHl75Z-Y  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Terminology and notation: input layer, hidden layer(s), output layer, with superscript [l] marking the layer and subscript i the unit within it. A single hidden layer already makes the net a two-layer network (input layer is not counted).
- layers numbered from input (a⁰ = x) to output (a² = ŷ)
- superscript [l] = layer number; subscript i = neuron number
- hidden units are "hidden" only in the sense of being neither input nor output

### Computing Neural Network Output
URL: https://www.youtube.com/watch?v=rMOdrD61IoU  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Ng shows a hidden-layer unit is just logistic regression repeated: z = wᵀx + b then a = σ(z), and the whole layer is the four equations Z[1] = W[1]x + b[1], A[1] = σ(Z[1]), Z[2] = W[2]A[1] + b[2], A[2] = σ(Z[2]). Stacking the per-node weight vectors vertically turns W[1] into a matrix, so the entire forward pass fits in four lines.
- each hidden node = one logistic regression (two steps)
- W[1] stacks the node weight vectors as rows: (n[1], n[0])
- full forward pass = four lines: two layers × (z then a)

### Vectorizing Across Multiple Examples
URL: https://www.youtube.com/watch?v=xy5MOQpx3aQ  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Building on the single-example equations, Ng stacks all m training examples as columns of X, turning Z[1] = W[1]X + b[1] into a matrix operation, with broadcasting again handling b. A[1], Z[2], A[2] follow the same column-stacked shape, so one forward pass covers the whole training set.
- X becomes (n⁰, m): examples as columns
- Z[1] = W[1]X + b[1] computes all hidden units for all examples at once
- A[2] ends up (1, m): the full prediction vector

### Gradient Descent For Neural Networks
URL: https://www.youtube.com/watch?v=7bLEWDZng_M  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
The full training recipe for a 2-layer net: initialize parameters, run the forward pass to get predictions and cost, compute gradients backward (the four key derivative formulas for dW[1], db[1], dW[2], db[2]), then update each parameter with gradient descent. The da/dz chain is exactly the chain rule from the computation graph.
- forward → cost → backward (gradients) → update, repeat
- the four gradient formulas (dZ[2], dW[2], db[2], dZ[1]…) follow the computation graph
- everything vectorizes over m examples

### Random Initialization
URL: https://www.youtube.com/watch?v=6by6Xas_Kho  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Why the network's weights cannot start at zero: every hidden unit would compute identical values and receive identical gradients (symmetry), so the layer never differentiates. The fix is small random values, e.g. W = np.random.randn(shape) * 0.01, while b may be zeroed.
- zero initialization → symmetric hidden units → no learning beyond one unit
- small random W breaks symmetry; b can stay 0
- the 0.01 factor keeps activations small near the linear regime of sigmoid

### Deep L-Layer Neural Network
URL: https://www.youtube.com/watch?v=2gw5tE2ziqA  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Ng lays out the general L-layer setting: layer counts n[l], parameter matrices W[l] and bias vectors b[l] for each layer, and the notation for activations a[l]. The forward and backward passes of a deep net are just the two-layer equations repeated L times, with careful shape bookkeeping (a[l] is (n[l], 1), W[l] is (n[l], n[l−1])).
- a[l] = σ(W[l]a[l−1] + b[l]) holds for every layer
- shapes: W[l] is (n[l], n[l−1]); b[l] is (n[l], 1)
- deep = many hidden layers; each adds abstraction/complexity, not new math

### Train/Dev/Test Sets
URL: https://www.youtube.com/watch?v=1waHlpKiNyY  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Practical data hygiene: train on the training set, choose among model variants on the dev (hold-out) set, and measure final performance once on the test set for an unbiased estimate. Modern big-data practice shrinks dev/test to 1–2% (e.g. 98/1/1 for a million examples), and dev/test should ideally come from the same distribution; if no unbiased estimate is needed, a test set is optional.
- dev set picks the model; test set gives one unbiased final score
- with huge data, small dev/test fractions (1%) beat the old 70/20/10
- keep dev and test from the same distribution whenever possible

### Softmax Regression
URL: https://www.youtube.com/watch?v=LLux1SW--oM  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
Softmax regression generalizes logistic regression to C classes: the output layer has C units, and the softmax activation turns the score vector z into a probability distribution — each unit's probability is e^(z_i) over the sum of all exponentials, so the outputs are non-negative and sum to 1. The class with the largest probability wins.
- softmax = "sigmoid for many classes", outputs sum to 1
- softmax is a smooth version of argmax: the winner is amplified, not hard-picked
- works with a one-hot target vector and cross-entropy loss

### Training Softmax Classifier
URL: https://www.youtube.com/watch?v=ueO_Ph0Pyqk  ·  Date: 2017-08-25 (YouTube upload)  ·  Status: summary
The training pass for a softmax classifier: forward pass with softmax in the last layer, loss = cross-entropy between prediction and the one-hot label (equivalent to negative log-likelihood), then backprop. The key simplification: the gradient of the loss with respect to z simplifies to (a − y), exactly the clean dz pattern of logistic regression, just vector-valued.
- loss = cross-entropy against the one-hot label; minimizing it = maximizing likelihood
- gradient dz = a − y generalizes the logistic case to vectors
- softmax + cross-entropy pair up for numerically stable, simple gradients

### Supervised vs. Unsupervised Machine Learning
URL: https://www.ibm.com/think/topics/supervised-vs-unsupervised-learning  ·  Date: unknown  ·  Status: summary
IBM's explainer contrasts the two families: supervised learning trains on labeled input-output pairs to predict outcomes (classification and regression), while unsupervised learning finds structure in unlabeled data (clustering, association, dimensionality reduction). It compares goals, applications, complexity, and drawbacks, then introduces semi-supervised learning as the middle ground for cases like medical imaging where labels are scarce but valuable.
- labeled data vs unlabeled data is the defining difference
- supervised: spam detection, sentiment, pricing; unsupervised: anomaly detection, recommendations, personas
- supervised is accurate but label-hungry; unsupervised scales but needs validation

### How would you explain neural networks to someone who knows very little about AI or neurology?
URL: https://www.quora.com/How-would-you-explain-neural-networks-to-someone-who-knows-very-little-about-AI-or-neurology/answer/Yohan-John  ·  Date: unknown  ·  Status: paywalled
Quora's answer page requires JavaScript to render, so the full text is not accessible. The question itself is famous: it collects lay-friendly explanations of neural networks for readers with no AI background.

### Using Neural Nets to Recognize Handwritten Digits
URL: http://neuralnetworksanddeeplearning.com/chap1.html  ·  Date: 2019-12 (publication date)  ·  Status: summary
Michael Nielsen's celebrated first chapter builds a 74-line MNIST digit recognizer from scratch. It walks through perceptrons (thresholded weighted sums), why perceptrons can't learn smoothly, sigmoid neurons as the fix, network architecture vocabulary (input/hidden/output layers), the quadratic cost, gradient descent, and the data-set setup — concluding with why more hidden layers can represent more complex features.
- perceptron: output flips 0/1 at a threshold, which blocks smooth learning
- sigmoid neuron = smoothed perceptron; small weight changes cause small output changes
- MNIST: 60,000 train / 10,000 test 28×28 greyscale images
- cost C = (1/2n) Σ ||y − a||²; gradient descent minimizes it over w and b

### Understanding Activation Functions in Neural Networks
URL: https://medium.com/the-theory-of-everything/understanding-activation-functions-in-neural-networks-9491262884e0  ·  Date: 2017-03-30 (publication date)  ·  Status: summary
A plain-language tour of activation functions and why they exist: a neuron computes a weighted sum that can be any real number, and the activation decides "how fired" it is. Step functions are too coarse for learning; linear activations collapse stacked layers back to one line; sigmoid/tanh give smooth, bounded, nonlinear output but suffer vanishing gradients; ReLU is cheap and sparse but can die (zero gradient for negative inputs, fixed by leaky variants).
- activation answers "how fired is this neuron?" given the raw sum
- all-linear layers collapse to a single linear layer — nonlinearity is required for depth
- sigmoid outputs (0,1) and suits classifiers; tanh is a shifted/scaled sigmoid
- ReLU = max(0, x): cheap and sparse, with the dying-ReLU caveat

### Loss function
URL: https://en.wikipedia.org/wiki/Loss_function  ·  Date: 2026-07-18 (last edited)  ·  Status: summary
Wikipedia's overview of loss/cost/error functions in optimization and decision theory: a function mapping outcomes to a real "cost", minimized by optimization. Covers the quadratic (squared-error) loss and its outlier sensitivity, the 0–1 loss used for classification accuracy, expected loss (risk) in frequentist and Bayesian views, and guidance for choosing a loss (continuous, differentiable).
- loss is minimized; objective function may be loss (min) or reward (max)
- quadratic loss is tractable and symmetric but dominated by outliers
- 0–1 loss counts misclassifications; smooth surrogate losses drive training

### Gradient descent
URL: https://en.wikipedia.org/wiki/Gradient_descent  ·  Date: 2026-08-07 (last edited)  ·  Status: summary
Wikipedia's treatment of gradient descent, the first-order iterative optimizer: repeatedly move against the gradient, x_{n+1} = x_n − η∇f(x_n), because the negative gradient is the steepest downhill direction. Discusses step-size (learning rate) choices, convexity (where all local minima are global), the foggy-mountain analogy, and stochastic gradient descent as the workhorse behind deep learning.
- negative gradient = direction of steepest decrease
- too-small step: slow; too-large step: divergence
- for convex f, gradient descent reaches the global minimum

### Calculus on Computational Graphs: Backpropagation
URL: http://colah.github.io/posts/2015-08-Backprop/  ·  Date: 2015-08-31 (publication date)  ·  Status: summary
colah's classic essay reframes backprop as reverse-mode differentiation on computational graphs. Naive path-summing explodes combinatorially; factoring the sums — merging paths at each node — makes both forward- and reverse-mode touch every edge exactly once. Reverse mode gets the derivative of one output w.r.t. all inputs in one pass, which is exactly what training needs: millions of parameters, one cost.
- backprop = reverse-mode differentiation = the chain rule, done efficiently
- naive path sums blow up; factoring at nodes fixes it
- one backward pass yields dJ/dW for every parameter — a million-fold speedup over forward mode

### Random Initialization For Neural Networks: A Thing Of The Past
URL: https://medium.com/data-science/random-initialization-for-neural-networks-a-thing-of-the-past-bfcdd806bf9e  ·  Date: 2018-02-25 (publication date)  ·  Status: summary
Aditya Ananthram compares three weight-init schemes on the same network: zero (no symmetry breaking — every neuron identical, accuracy ≈ random), small random values (W = np.random.randn(shape) * 0.01, breaks symmetry, accuracy 0.83), and He initialization (W = np.random.randn(shape) * sqrt(2/n_prev), accuracy 0.96). The lesson: initialization choice measurably shapes convergence speed and final accuracy.
- zero weights → no symmetry breaking → the net degenerates to one neuron
- random small weights break symmetry; scale matters
- He init scales by sqrt(2/n_prev) — why depth was once thought to need "special" starts

### Initialization of deep networks
URL: https://aiml.com/weight-initialization-in-deep-neural-networks/  ·  Date: 2025-12-17 (updated)  ·  Status: summary
AIML.com's guide explains why initialization matters: too-large weights make activations and gradients explode with depth, too-small ones make them vanish. It walks through classical schemes (random normal/uniform), variance-preserving modern ones — Xavier/Glorot (variance 2/(d_in + d_out), for tanh/sigmoid), He/Kaiming (variance 2/d_in, for ReLU) — plus orthogonal and LeCun, with an MNIST experiment showing He converging fastest, Xavier close behind, and zero init stalling entirely.
- goal: keep activation and gradient variance roughly constant across layers
- Xavier suits tanh/sigmoid; He compensates for ReLU zeroing half its inputs
- rule of thumb: match the initializer to the activation function

### Multiclass classification
URL: https://en.wikipedia.org/wiki/Multiclass_classification  ·  Date: 2026-04-28 (last edited)  ·  Status: summary
Wikipedia defines multiclass (multinomial) classification as assigning instances to one of three or more classes, distinct from multi-label classification where an instance gets several labels. Some algorithms (neural networks, multinomial logistic regression) handle many classes natively, while inherently binary ones (SVM) need decomposition strategies like one-vs-rest or one-vs-one; the page also analyzes confusion-matrix criteria for doing better than chance.
- multiclass: one class per instance; multi-label: many labels per instance
- neural nets and softmax regression extend naturally to K classes
- binary-only algorithms decompose via one-vs-rest / one-vs-one

### Derivation: Derivatives for Common Neural Network Activation Functions
URL: https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions  ·  Date: 2020-06-29 (publication date)  ·  Status: summary
Dustin Stansbury derives, step by step, the derivatives backprop needs: identity (g′ = 1), logistic sigmoid (g′ = g(1 − g)), and tanh (g′ = 1 − g²). The punchline is the caching trick: each derivative is a simple function of the already-computed feed-forward activation, so gradients cost a multiply/subtract instead of a re-evaluation of the exponential.
- sigmoid: σ′(z) = σ(z)(1 − σ(z)) via the quotient rule and a ±1 trick
- tanh: tanh′(z) = 1 − tanh²(z)
- derivatives reuse the forward activations — no extra exponentiation in backprop

### What is One Hot Encoding?
URL: https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f  ·  Date: 2017-08-03 (publication date)  ·  Status: summary
Vasudev explains why categorical labels must be binarized: label-encoding ("VW=1, Acura=2, Honda=3") silently imposes an ordering the model will exploit, e.g. averaging VW and Honda "equals" Acura. One-hot encoding replaces one categorical column with K binary columns (is_daffodil, is_lily, is_rose), each 0/1, killing the fake ordering.
- label encoding implies rank among categories — a recipe for wrong averages
- one-hot = one binary feature per category, exactly one 1 per row
- in classification: convert integer labels to one-hot vectors to match softmax output

### Softmax function
URL: https://en.wikipedia.org/wiki/Softmax_function  ·  Date: 2026-06-30 (last edited)  ·  Status: summary
Wikipedia's definition of softmax as the normalized exponential that maps a tuple of K reals to a probability distribution (each in (0,1), summing to 1), generalizing the logistic function to many dimensions. It covers the temperature parameter, softmax as a smooth argmax, its statistical-mechanics form (Boltzmann distribution), the identity for its Jacobian (∂σ_i/∂z_j = σ_i(δ_ij − σ_j)), and its role as the last activation of classification networks.
- σ(z)_i = e^(z_i) / Σ_j e^(z_j); outputs sum to 1
- exponentials amplify the largest input — a smooth, differentiable argmax
- gradient w.r.t. z takes the convenient form σ_i(δ_ij − σ_j)

### What is the intuition behind SoftMax function?
URL: https://www.quora.com/What-is-the-intuition-behind-SoftMax-function  ·  Date: unknown  ·  Status: paywalled
Quora blocks unauthenticated readers with a JavaScript gate, so the answer text is not retrievable. The thread's theme is the standard intuition: softmax converts a vector of scores into probabilities, sharply favoring the winner while remaining differentiable.

### Cross entropy
URL: https://en.wikipedia.org/wiki/Cross-entropy  ·  Date: 2026-08-04 (last edited)  ·  Status: summary
Wikipedia defines cross-entropy H(p, q) = −E_p[log q] as the average number of bits needed to identify an event when the coding is optimized for q rather than the true p; it equals H(p) + KL(p‖q). Crucially for this project: maximizing likelihood is equivalent to minimizing cross-entropy, and the binary-classification cross-entropy loss −(y log ŷ + (1−y) log(1−ŷ)) — also called log loss — is exactly what logistic regression minimizes, with gradient Xᵀ(ŷ − y).
- H(p,q) = −Σ p(x) log q(x); minimized when p = q
- MLE ⇔ minimizing cross-entropy (log-likelihood = −N·H)
- binary cross-entropy loss doubles as the logistic regression cost

### Loss Functions: Cross-Entropy
URL: https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html  ·  Date: unknown  ·  Status: summary
The ML Cheatsheet's cross-entropy section: the log loss measures how far a predicted probability is from the true label, rising steeply for confident-and-wrong predictions, with a perfect model scoring 0. Binary form −(y log p + (1−y) log(1−p)), multiclass form −Σ_c y_o,c log p_o,c (summed over classes per observation).
- log loss: predicted probability diverging from the label → larger loss
- especially punishes confident wrong predictions
- multiclass: sum over classes of −y·log(p) per observation

### What is Pickle in python?
URL: https://yasoob.me/2013/08/02/what-is-pickle-in-python/  ·  Date: 2013-08-02 (publication date)  ·  Status: summary
Yasoob's intro to pickle, Python's serialization module: pickle.dump writes an object to a file as a byte stream, pickle.load reconstructs it later or in another script. The post lists use cases — saving program state, sending objects over TCP, storing in databases, caching — and notes files must be opened in binary mode ('wb'/'rb').
- pickle serializes arbitrary Python objects into a byte stream
- dump to save, load to restore; both operate on file objects
- open files in binary mode, 'wb' to write, 'rb' to read

### Predictive analytics
URL: https://en.wikipedia.org/wiki/Predictive_analytics  ·  Date: 2026-07-19 (last edited)  ·  Status: summary
Wikipedia surveys predictive analytics: statistical, data-mining and machine-learning techniques that analyze current and historical facts to predict unknown events. Covers the core pipeline (define objectives, analyze source data, build and validate models, deploy, monitor), main technique families (regression, time-series like ARIMA, machine learning), and business applications from marketing to underwriting.
- predictive analytics = forecasting unknown outcomes from historical data
- pipelines: objectives → data → model → validate → deploy → maintain
- classification (this project) is one of its core technique families

### Maximum Likelihood Estimation
URL: https://medium.com/data-science/maximum-likelihood-estimation-984af2dcfcac  ·  Date: 2019-02-03 (publication date)  ·  Status: summary
William Fleshman's MLE tutorial: write a probabilistic model of how the data was generated, then pick the parameters that make the observed data most likely. Because products of small probabilities underflow, MLE maximizes the log-likelihood (a sum, thanks to logs), and the coin-flip example derives the natural estimate p̂ = heads/flips.
- MLE: choose θ maximizing the likelihood of the observed data
- log-likelihood turns products into sums and keeps numerics stable
- for a Bernoulli coin, MLE gives p̂ = h/n — the intuitive frequency estimate

## References

- numpy.zeros — https://numpy.org/doc/stable/reference/generated/numpy.zeros.html
- numpy.random.randn — https://numpy.org/doc/stable/reference/generated/numpy.random.randn.html
- numpy.exp — https://numpy.org/doc/stable/reference/generated/numpy.exp.html
- numpy.log — https://numpy.org/doc/stable/reference/generated/numpy.log.html
- numpy.sqrt — https://numpy.org/doc/stable/reference/generated/numpy.sqrt.html
- numpy.where — https://numpy.org/doc/stable/reference/generated/numpy.where.html
- numpy.max — https://numpy.org/doc/stable/reference/generated/numpy.max.html
- numpy.sum — https://numpy.org/doc/stable/reference/generated/numpy.sum.html
- numpy.argmax — https://numpy.org/doc/stable/reference/generated/numpy.argmax.html
- pickle / pickle.dump / pickle.load — https://docs.python.org/3/library/pickle.html

## Quiz Hooks

- neuron/perceptron — a unit that takes weighted inputs plus a bias and produces an output through an activation
- forward propagation — passing data through the layers to produce a prediction
- activation function — nonlinear function applied to the weighted sum that decides "how fired" a neuron is
- loss/cost function — measure of how wrong the prediction is; cost averages loss over the training set
- gradient descent — iterative algorithm stepping opposite the gradient to minimize the cost
- backpropagation — chain rule applied backward through the network to compute gradients of the cost
- vectorization — replacing explicit loops with numpy array operations for speed
- one-hot encoding — representing a class as a vector with a single 1 at its position
- softmax — normalized exponential that turns a score vector into a probability distribution summing to 1
- cross-entropy — −Σ p log q; the loss minimized when training a classifier, equivalent to negative log-likelihood
- logistic regression — binary classifier with output σ(wᵀx + b), trained with cross-entropy loss
- bias term — scalar added to the weighted sum; shifts the decision boundary
- random initialization — starting weights as small random values to break symmetry between neurons
- train/dev/test split — train on the first, pick the model on dev, measure final performance once on test
- sigmoid — σ(z) = 1/(1 + e^(−z)), squashing any real number into (0, 1)
- MLE — maximum likelihood estimation: pick parameters that make observed data most probable
- pickle — Python module that serializes objects to bytes and back (dump/load)
