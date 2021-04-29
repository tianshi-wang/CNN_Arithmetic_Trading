# https://usmanr149.github.io/urmlblog/cnn/2020/05/01/Salincy-Maps.html
import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
import matplotlib.pyplot as plt


def salincy_map(image_path, model):
    _img = keras.preprocessing.image.load_img(image_path,target_size=(224,224))
    image_name = image_path.split('/')[-1].split('.')[-2]
    plt.imshow(_img)
    plt.show()

    img = keras.preprocessing.image.img_to_array(_img)
    img = img.reshape((1, *img.shape))
    print('image reshaped ')
    y_pred = model.predict(img)
    print('predicted as ')
    print(y_pred)

    images = tf.Variable(img, dtype=float)

    with tf.GradientTape() as tape:
        pred = model(images, training=False)
        class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
        loss = pred[0][class_idxs_sorted[0]]  # we need to change it to regression predictions
    
    grads = tape.gradient(loss, images)
    dgrad_abs = tf.math.abs(grads)
    dgrad_max_ = np.max(dgrad_abs, axis=3)[0]

    arr_min, arr_max  = np.min(dgrad_max_), np.max(dgrad_max_)
    grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)

    fig, axes = plt.subplots(1,2,figsize=(14,5))
    axes[0].imshow(_img)
    i = axes[1].imshow(grad_eval,cmap="jet",alpha=0.8)
    fig.colorbar(i)
    fig.savefig(f'{image_name}_salincy.jpg')

if __name__ == '__main__':
    IMAGE_PATH = './cat_front.jpeg'
    model = keras.applications.VGG16(weights='imagenet')
    salincy_map(IMAGE_PATH, model)
