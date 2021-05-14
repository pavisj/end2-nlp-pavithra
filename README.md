# end2-nlp-pavithra

### Session 1 - Basics
##### What is a neural network neuron?

A DNN or Neural network neuron is a storage unit that stores a number or a "signal". Our neurons in our brain in addition to having a storage unit, also have their own computation unit. A neural network neuron has the computation unit outside it along with a weight (a number) for each neuron. A NN neuron stores the result of the computation of its input based on the weight and the activation function

$$ z (neuron_a) =  \text{Activation function} * (\text {input from the input connection} * \text{weight of the neuron}) + bias)$$

The output of a $ neuron_a $ serves as an input for the $ neuron_b $ in the next layer. The two neurons $ neuron_a \text{and } neuron_b $ are connected by the outgoing connection from $ neuron_a$


##### What is the use of the learning rate?

In gradient descent, the learning rate handles magnitude with which a weight changes according to the loss value of the NN. While a gradient sets the direction in which the NN weights are decreasing wrto the loss value, such that local loss minima is achieved, the learning rate ensures the smooth convergence of the loss optimization at a steady rate.Larger learning rate might lead to losing the local minima and will not lead to gradual reduction in ths loss, thereby hindering convergence (loss reduction as it approaches a local minima)

##### How are weights initialized?

Weights are initialized based on Gaussian or normal distribution (random-normal)  with zero-mean and a calculated variance (smaller weights). The intuition behind the generalization of a NN is about learning from the input distribution. The weights should follow a normal distribution (or something similar) such that as it learns during the training it is able to converge better. The scale of the initial weight distribution affects the convergence of the network. The smaller the scale, better the convergence.

Using weights as constant values hinders convergence mainly because of vanishing/exploding gradients. 

##### What is "loss" in a neural network?

The difference between the NN output and the ground truth output. The loss function determines the learning (training) outcome in a NN. A loss function shows the NN the gaps that it needs to fill/learn, such that it is able to perform a particular task.

##### What is the "chain rule" in gradient flow?

Chain rule helps in find the partial derivative of a function A  with respect to another function B, by using a function C

![Image](./images/chain-rule-1.png)

Chain rule helps in propagating the loss value from one layer to another such that a given NN's weight is changed wrto to the loss value

![image](./images/propagation.png)
![image](./images/propagation-gif.gif)
![image](./images/propagation-2.png)

Source: https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c

### Session 2
#### Excelling in Backpropagations in Excel 
*(contextualized pun intended)*

Here is a Neural network trained in Excel (or in my case MacOS Numbers) with a learning rate of 0.5

![image](./images/the-complete-picture-learning-rate-0-5.png)

Read on to know how this was done. The excel sheet also have been attached to this: https://github.com/pavisj/end2-nlp-pavithra/blob/main/session2-backpropagation.xlsx

**Step 1:**
Create the Neural network with the 8 weights - w1 to w8. It takes two inputs i1 and i2 and gives two outputs o1 and o2. L2 loss is used for defining the loss.

![image](./images/backprop-1.png)

**Step 2**
Write the formulae for the forward pass of the Neural network

![image](./images/backprop-2.png)

**Step 3**
Write all the input and output variables

![image](./images/backprop-3.png)


![image](./images/backprop-4.png)

![image](./images/backprop-6.png)

![image](./images/backprop-7.png)

![image](./images/backprop-8.png)

![image](./images/backprop-10.png)

![image](./images/backprop-11.png)

![image](./images/backprop-12.png)

![image](./images/backprop-13.png)

![image](./images/backprop-15.png)

![image](./images/learning_rate_0_1.png)
![image](./images/learning_rate_0_2.png)
![image](./images/learning_rate_0_5.png)
![image](./images/learning_rate_0_8.png)
![image](./images/learning_rate_1_0.png)
![image](./images/learning_rate_2_0.png)



