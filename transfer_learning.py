# https://machinelearningmastery.com/how-to-use-transfer-learning-when-developing-convolutional-neural-network-models/
# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5

from pathlib import Path
import os
import pandas as pd
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow import keras

def get_data(ticker):
    data_folder = Path(__file__).parent/"training_data"
    arrays = []
    folder = data_folder/ticker
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".npy"):
            arrays.append(np.load(folder/filename))
    feature = np.array(arrays)
    feature = feature.transpose(0, 2, 3, 1)
    labels = pd.read_csv(folder/"labels.csv")['label'].values
    return feature, labels
    
def transfer_learn(ticker):
    feature, labels = get_data(ticker)
    model = keras.models.load_model('./test_model')
    for layer in model.layers[:-1]:   # edit the trainable layers here
        layer.trainable = False
    print(model.summary())
    model.compile(optimizer='adam',
              loss=tf.keras.losses.mae,
              metrics=['accuracy'])

    train = feature[:280, ...]
    test = feature[280:, ...]
    train_labels = labels[:280, ...]
    test_labels = labels[280:, ...]

    history = model.fit(train, train_labels, epochs=10)
    test_loss, test_acc = model.evaluate(test,  test_labels, verbose=2)
    predicts = model.predict(test)
    comparison = pd.DataFrame({'predicts': predicts.flatten(), 'labels': test_labels})
    return predicts


# def vgg_transfer(ticker):
#     x, y = get_data(ticker)
#     model = VGG16(include_top=False, input_shape=(300, 300, 3))  # must have 3 channels, can have different w and h
#     # add new classifier layers
#     flat1 = Flatten()(model.layers[-1].output)
#     class1 = Dense(1024, activation='relu')(flat1)
#     output = Dense(10, activation='softmax')(class1)
#     # define new model
#     model = Model(inputs=model.inputs, outputs=output)
#     # summarize
#     model.summary()

if __name__ == '__main__':
    tickers = ['AAPL', 'QQQ', 'AMZN', 'MSFT', 'INTC', 'MARA']
    for ticker in tickers:
        predicts = transfer_learn(ticker)
