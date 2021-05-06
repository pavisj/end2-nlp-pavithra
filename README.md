# end2-nlp-pavithra


##### What is a neural network neuron?

A DNN or Neural network neuron is a storage unit that stores a number or a "signal". Our neurons in our brain in addition to having a storage unit, also have their own computation unit. A neural network neuron has the computation unit outside it along with a weight (a number) for each neuron. A NN neuron stores the result of the computation of its input based on the weight and the activation function

$$ z (neuron_a) =  \text{Activation function} * (\text {input from the input connection} * \text{weight of the neuron}) + bias)$$

The output of a $ neuron_a $ serves as an input for the $ neuron_b $ in the next layer. The two neurons $ neuron_a \text{and } neuron_b $ are connected by the outgoing connection from $ neuron_a$


##### What is the use of the learning rate?

In gradient descent, the learning rate handles magnitude with which a weight changes according to the loss value of the NN. While a gradient sets the direction in which the NN weights are decreasing wrto the loss value, such that local loss minima is achieved, the learning rate ensures the smooth convergence of the loss optimization at a steady rate.

##### How are weights initialized?

Weights are initialized based on Gaussian or normal distribution with zero-mean and a calculated variance. The intuition behind the generalization of a NN is about learning from the input distribution. The weights should follow a normal distribution (or something similar) such that as it learns during the training it is able to converge better and 'mimick' the outcome of the input distribution. 

Using weights as constant values hinders convergence mainly because of vanishing/exploding gradients. It is not primed to learn from the input distribution and generalize.

##### What is "loss" in a neural network?

The difference between the NN output and the ground truth output. The loss function determines the learning (training) outcome in a NN. A loss function shows the NN the gaps that it needs to fill/learn, such that it is able to perform a particular task as indicated by the Ground Truth.

##### What is the "chain rule" in gradient flow?

Chain rule helps in find the partial derivative of a function A  with respect to another function B, by using a function C
![Image](./images/chain-rule-1.png)
Chain rule helps in propagating the loss value from one layer to another such that a given NN's weight is changed wrto to the loss value
![image](./images/propagation.png)
![image](./images/propagation-gif.gif)
![image](./images/propagation-2.png)
