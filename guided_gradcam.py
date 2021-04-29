import cv2
import numpy as np
import tensorflow as tf

# can get the position in 2d heatmap

def guided_gradcam(image_path, layer_name, model):
    IMAGE_PATH = image_path
    LAYER_NAME = layer_name
    CAT_CLASS_INDEX = 281  # we don't need it
    image_name = image_path.split('/')[-1].split('.')[-2]

    img = tf.keras.preprocessing.image.load_img(IMAGE_PATH, target_size=(224, 224))
    img = tf.keras.preprocessing.image.img_to_array(img)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(LAYER_NAME).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, CAT_CLASS_INDEX]  # we only need prediction of regression

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    # print(loss.numpy().shape)
    # print(conv_outputs.numpy().shape)

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads
    # print(guided_grads.numpy().shape)

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    # print(weights.numpy().shape)

    cam = np.ones(output.shape[0: 2], dtype = np.float32)
    # print(cam.shape)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 0.5, cam, 1, 0)
    cv2.imwrite(f'{image_name}_guidedcam.png', output_image)

if __name__ == '__main__':
    IMAGE_PATH = './cat_front.jpeg'
    LAYER_NAME = 'block5_conv3'
    model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
    guided_gradcam(IMAGE_PATH, LAYER_NAME, model)


