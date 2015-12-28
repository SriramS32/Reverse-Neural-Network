# Reverse-Neural-Network

Hello!

**Background:** I started building this project from scratch in March when I was interested in coding neural network-based computer vision. Eventually, I wanted to create a neural network that could produce images rather than just recognizing images. In terms of computer vision, this meant asking for the input pixels of a specific image that the network has recognized. The progression of work is in the project named ReverseNNDevelopment.

**Repository** contains
- CreativeArtColor.java contains the main neural network code that implements the creation of input functionality
- CreativeArtColor.class is a compiled version of the java file
- Sample image set of image0.jpg...image3.jpg, combining Starry Night with The Scream with white and black as control backgrounds.
- weightsColor.txt is a sample weights file, if you would like to run and reverse immediately without training (which takes the longest time)
The default parameters in the java file are set for these images, with a quarter million inputs per image (300 by 300)

**Repository** requires
- A data set composing of images (in the format of image0...imagen)
- Updates of parameters such as changing input, hidden, and output nodes based on the size of image or complexity of network. Also, will need to update the number of cases based on the number of images in image set.

An example of usage:
```javascript
//In command line, after downloading all files in one directory and cd into that directory
javac CreativeArtColor.java //to compile file and create CreativeArtColor.class

java CreativeArtColor //runs program
train 	//will first train the network based on image set, epoch errors will be printed out

java CreativeArtColor
run 	//will evaluate the image set with trained network

java CreativeArtColor
reverse //will reverse network and create an image

//Note: Optimization of parameters can be performed, runs faster for me on blueJ rather than terminal, may want to change accepted error to prune clarity of image
```
An example output image after reversing looks like this (synthesis of The Scream and Starry Night):

![id](https://raw.githubusercontent.com/SriramS32/Reverse-Neural-Network/master/ExampleSynthesis.jpg)

#Explanation of Theory
The neural network is a simple multilayer perception that uses a logistic activation function and stochastically updates the weights. The network has 3 options:
- **train:** A new network is trained over a data set, the weights of the neural network are saved to a text file.
- **run:** Networks can be created from weight files and run on test images.
- **reverse:** The weight file also allows the user to reverse back images. A new network is created, and the images are created by using multivariable calculations and minimizing the error function with respect to the inputs rather than the weights (fixed after training). This will train the inputs to fit the desired output and over training cycles, the computer will draw its computational ideals.

Back-propagation is implemented in both training and reversing.

Common application to Synthesis: A value such as a 1 is assigned to any piece of art. A 0 is assigned to background images such as a square of white and black. When the program is asked to train the inputs to achieve an output of 1, it will appear to draw a synthesized artwork image.

For more information, see website.

#Future Plans
Currently the code is only applied to a small database of images, but it could easily expand to other realms of neural network application. The math could be reapplied to other networks such as CNNs or RNNs and phoneme analysis and pattern identification beyond computer vision. Other patterns could be explored besides synthesis.

#License
BSD

Sriram Somasundaram :)
