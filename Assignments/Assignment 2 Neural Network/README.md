The purpose of this homework is to allow you to obtain deeper understanding of the underlying working mechanisms and theory behind neural networks. The homework consists of a series of tasks which allow you to understand, develop or re-implement some of the features of the neural networks.

__Task 1:__
Read and fully understand the article Nothing but NumPy: Understanding & Creating Neural Networks with Computational Graphs from Scratch [Link](https://www.kdnuggets.com/2019/08/numpy-neural-networks-computational-graphs.html). If you have not programmed in Python before, please type yourself all the code provided at the end of the article. This will help you get a feel of Python programming and help you understand what is going on. (Even if you programmed in Python before I strongly recommend you do not skip this step.)

__Task 2:__
Get the program from Task 1 to work on the MNIST database [Link](http://yann.lecun.com/exdb/mnist/). Evaluate the performance (classification error) of your program in comparison to others [Link](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html). Use minimum 3 internal layers with minimum 20 activation nodes in each internal layer.

Save your program and the corresponding results as the version 1 (V1).

__Task 3:__

Change the following in your program:

Replace batch gradient descent with mini-batch gradient descent [Link](http://cs231n.github.io/optimization-1/#gd) to train the network. Re-run your program on the MNIST dataset and compare the performance (execution speed due to mini-batch change and the classification error). Save this program as version 2 (V2).

Change the activation function to another based on your choice [Link](https://en.wikipedia.org/wiki/Activation_function). Note that changing the activation function will require you to change backpropagation derivatives. Re-run your program on the MNIST dataset and compare the performance (execution speed due to the activation function change and the classification error). Save this program as version 3 (V3).


__Task 4:__
Summarize and explain the observed performance changes in all 3 versions of your program in a short presentation (not more than 10 slides).


__Submission__

Submit all versions of your program, results, and slides in a zipped directory named _COSC606\_HW2\_YourFirstAndLastName.zip_. This homework is to be done individually. You are allowed to discuss, brainstorm, and consult each other with respect to any issues or challenges, but all programming must be done individually.