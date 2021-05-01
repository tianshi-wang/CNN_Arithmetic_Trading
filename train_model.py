import os

import yfinance as yf
import pandas as pd
import numpy as np
from pyts.image import GramianAngularField
from pathlib import Path

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt

# all tickers to pull data
tickers = ['AAPL', 'QQQ', 'AMZN', 'MSFT', 'INTC', 'MARA']
data_folder = Path(__file__).parent/"training_data"


# load data
arrays = []
folder = data_folder/'AAPL'
for filename in os.listdir(folder):
    if filename.endswith(".npy"):
        arrays.append(np.load(folder/filename))

data = np.array(arrays)
data = data.transpose(0, 2, 3, 1)
labels = pd.read_csv(folder/"labels.csv")['label'].values

# Train CNN model
model = CNN()
predict, test_labels = model.train(data, labels)

comparison = pd.DataFrame({'predicts': predicts.flatten(), 'labels': test_labels})

trainable_count = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_count = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))

print(model.history)