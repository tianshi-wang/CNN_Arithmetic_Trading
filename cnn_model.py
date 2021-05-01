import tensorflow as tf
from ten.keras import layers,models

from sklearn.model_selection import train_test_split

class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()

        self.layers = models.Sequential(
            layers.Conv2d(layers.Conv2D(1, (3, 3), activation='relu', input_shape=(20, 20, 2), )),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(1, (2, 2), activation='relu'),
            layers.MaxPooling2D((3, 3)),

            layers.Flatten(),
            layers.Dense(3),
            layers.Dense(1),
        )
        self.history = None
        self.test_loss = None
        self.test_acc = None
        self.trainable_weights = None
        self.non_trainable_weights = None

    def train(self, X, y):
        predict = None
        test_labels = None

        # split train and validation data with the size of 0.2
        train, test, train_labels, test_labels = train_test_split(X, y, test_size=0.2, random_state=13)

        self.layers.compile(optimizer='adam', loss=tf.keras.losses.mae, metrics=['accuracy'])
        self.history = self.layers.fit(train, train_labels, epochs=10,
                    # validation_data=(test_images, test_labels)
                    )
        self.test_loss, self.test_acc = model.evaluate(test,  test_labels, verbose=2)
        
        self.predict = self.layers.predict(test)
        self.trainable_weights = self.layers.trainable_weights
        self.non_trainable_weights = self.layers.non_trainable_weights

        return predict, test_labels