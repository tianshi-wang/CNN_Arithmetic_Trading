import os

import yfinance as yf
import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


# all tickers to pull data
tickers = ['AAPL', 'QQQ', 'AMZN', 'MSFT', 'INTC', 'MARA']
data_folder = Path(__file__).parent/"training_data"

arrays = []
folder = data_folder/'AAPL'
for filename in sorted(os.listdir(folder)):
    if filename.endswith(".npy"):
        arrays.append(np.load(folder/filename))

feature = np.array(arrays)
feature = feature.transpose(0, 2, 3, 1)
labels = pd.read_csv(folder/"labels.csv")['label'].values

train = feature[:1000, ...]
test = feature[1000:, ...]

train_labels = labels[:1000, ...]
test_labels = labels[1000:, ...]

model = models.Sequential()
model.add(layers.Conv2D(2, (2, 2), activation='sigmoid', input_shape=(20, 20, 7), ))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())
model.add(layers.Conv2D(2, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())
# model.add(layers.Conv2D(2, (2, 2), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.BatchNormalization())


# model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
# model.add(layers.Dense(3))
model.add(layers.Dense(1))

model.compile(optimizer='adam',
              loss=tf.keras.losses.mae,
              metrics=['accuracy'])

history = model.fit(train, train_labels, epochs=10,
                    # validation_data=(test_images, test_labels)
                    )
test_loss, test_acc = model.evaluate(test,  test_labels, verbose=2)
predicts = model.predict(test)
comparison = pd.DataFrame({'predicts': predicts.flatten(), 'labels': test_labels})

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))
print(history)

