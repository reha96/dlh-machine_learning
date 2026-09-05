Write a blog post explaining the mechanics, pros, and cons of the following optimization techniques:

    Feature Scaling
    Batch normalization
    Mini-batch gradient descent
    Gradient descent with momentum
    RMSProp optimization
    Adam optimization
    Learning rate decay

To start, all the techniques that will be discussed concern forward pass (
    or is it backpropagation, or both?)  

feature scaling, works either by standardization (X-x_mean)/(x_sigma),
this results in mean 0 and variance 1 for input var. a second option is
by normalization ((X*gamma)+beta), which typically squeezes variable between
0 and 1, or any specified range. purpose is to avoid explosing the gradient
after many many passes

batch normalization, we adjust gradient/learning rate only at certain 
intervals, to avoid over adapting our descent

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

