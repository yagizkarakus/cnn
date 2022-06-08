# Layers

Simple convolutional neural network which use only convolution layer max pooling layer and softmax layer.
## Convolution Layer
  apply random filter to the image and seperate the original image to regions(which are equal to filter size)
  and forward propogate to  next layer
## Max Pool Layer
  apply max pooling to the regions and forward propogate to  next layer
 ## Softmax Layer
  flatten the image and compare the predictions with original imag labels if it doesn't match start a back prop chain 
  through the first layer and adjust the weights of edges
# Dataset
The code train mnist data set. 
