# CNN-based Arithmetic Trading Model using Time-Series Encoding and Transfer Learning

It is challenging to train deep learning models using stock time series data e.g. prices and volumes for  for two reasons.

Firstly, it is difficult to encode the time series data into a format a deep learning model can recognize the pattern within it. 

Secondly, the number of training examples usually is too small to train a complex deep learning model.

In this work, we encode each time-series data into a 20*20*7 image. Each channel is constructed using
different market data or encoding method in order to expose
the recognizable patterns for the models. We use transfer
learning to overcome the small number of training data
for each stock leveraging the similarity between different
stocks. Specifically, we firstly train our convolutional neural
network (CNN) model on all stocks, then fix the weights
of all the layers except the last one and fine-tune the model
for each stock. With the trained model, we proposed several
different trading strategies and test their performance. We
compared the results with the benchmark strategy and discussed
the strength and limitation of the model. Finally, we
visualize the model to understand the learned patterns.

![Input and saliency map for each channel of an encoded image]( https://github.com/tianshi-wang/CNN_Arithmetic_Trading/blob/main/image.jpg?raw=true)




