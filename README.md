# Reverse-Neural-Network

Hello!

**Background** I started building this project in March on my computer while learning about how to code neural network based computer vision from scratch.
Eventually, I wanted to create a neural network that could reverse back inputs. In terms of computer vision, this meant asking for a specific image that the network has recognized. The progression of work is in the project named ReverseNNDevelopment.

**Repository contains
- CreativeArtColor.java which is main neural network code that implements the creation of input functionality

#Explanation of Theory
The neural network is a simple multilayer perception that uses a logistic activation function and stochastically updates the weights.
After training over a data set, the weights of the neural network are saved to a text file. Networks can be created from weight
files and run on test images.  The weight file also allows the user to reverse back images. The images are created by using
multivariable calculations and minimizing the error function with respect to the inputs rather than the weights (fixed after training).

Back-propagation is implemented in both training and reversing.


An example of usage:
```javascript
//In command line
javac CreativeArtColor.java //to compile file and create CreativeArtColor.class
java CreativeArtColor //runs program
```

#Future Plans