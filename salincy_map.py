# https://usmanr149.github.io/urmlblog/cnn/2020/05/01/Salincy-Maps.html
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os

tickers = ['AAPL', 'QQQ', 'AMZN', 'MSFT', 'INTC', 'MARA']

def get_data(ticker):
    data_folder = Path(__file__).parent/"training_data"
    arrays = []
    folder = data_folder/ticker
    for filename in os.listdir(folder):
        if filename.endswith(".npy"):
            arrays.append(np.load(folder/filename))
    feature = np.array(arrays)
    feature = feature.transpose(0, 2, 3, 1)
    labels = pd.read_csv(folder/"labels.csv")['label'].values
    return feature[200:, ...], labels[200:, ...]

def get_min_max(feature, labels, ticker):
    predicts = model.predict(feature)[:,0]
    compare = np.abs(predicts - labels) / (np.abs(labels)+ 1e-8)
    min_dif = feature[np.argmin(compare)]
    max_dif = feature[np.argmax(compare)]
    return min_dif, max_dif

def salincy_map(img, model, image_name):
    img = img.reshape((1, *img.shape))
    y_pred = model.predict(img)
    images = tf.Variable(img, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0] # we need to change it to regression predictions
    
    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)

    curt_min, curt_max = np.min(dgrad_abs.numpy()), np.max(dgrad_abs.numpy())
    for i in range(dgrad_abs.numpy().shape[3]):
        curt_layer = dgrad_abs[0,:,:,i]
        curt_grad_eval = (curt_layer - curt_min) / (curt_max - curt_min + 1e-18)
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        mat = axes[0].matshow(curt_layer, cmap='rainbow', origin='lower')
        fig.colorbar(mat, ax=axes[0])
        sal= axes[1].imshow(curt_grad_eval,cmap="jet",alpha=0.8)
        fig.colorbar(sal, ax=axes[1])
        fig.savefig(f'{image_name}_{str(i)}_salincy.jpg')

if __name__ == '__main__':
    model = keras.models.load_model('./test_model')
    for ticker in tickers:
        feature, labels = get_data(ticker)
        min_dif, max_dif = get_min_max(feature, labels, model)
        salincy_map(min_dif, model, f'{ticker}_min')
        salincy_map(max_dif, model, f'{ticker}_max')

