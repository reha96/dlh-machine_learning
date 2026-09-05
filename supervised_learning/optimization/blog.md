Write a blog post explaining the mechanics, pros, and cons of the following optimization techniques:

    Feature Scaling
    Batch normalization
    Mini-batch gradient descent
    Gradient descent with momentum
    RMSProp optimization
    Adam optimization
    Learning rate decay

To start, all the techniques that we discuss concern forward pass (
    or is it backpropagation, or both?), and aim to find the optimal 
    Weights (W) and biases (b) for the neural network model.


Feature scaling has many variants and concerns the mathematical range of input
 data's features (not labels or outcome variable). It is thus a data prepocessing technique.
In most cases, we achieve scaling by standardization (X-x_mean)/(x_sigma),
this results in mean 0 and variance 1 for input var. a second option is
by min/max normalization ((X*gamma)+beta), which typically squeezes variable between
0 and 1, or any specified range. purpose is to avoid explosing the gradient
after many many passes. Simply put, gradient descent loss minimization works
better when all input features are of same scale.

batch normalization, is a network-level regularization technique. Remember that 
regularization techniques we saw so far were Lasso (L1) and Ridge (L2), with their
key objective being the selection of weights that best explain/capture the variance in output variable, while penalizing overfitting. Simpler, choosing the minimal set of weights to explain our outcome variable.  Mechanically, this technique normalizes input features at every layer of the network so each layer works with the same standardized distribution of weights. It allows 
for larger learning rates (for what? for gradient descent?)

Mini-batch gradient descent which does exactly the above

Gradient descent with momentum, exponential weights allow a large weight
for past step, while ignoring more and more earlier steps. this way past 
does not matter after a while, while current and last steps matter 
relatively more

RMSProp optimization, adjust variable (which one?) by its mean and variance,
ie its 1st and 2nd moments

Adam optimization combines RMS and momentum, and applies a correction for their bias for 0

Learning rate decay, allows us to adjust dynamically for the learning rate,
similar to momentum idea, where rate gradually decreases (ideally)

