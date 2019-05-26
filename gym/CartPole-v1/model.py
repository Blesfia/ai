import tensorflow as tf
import numpy as np

class Model:
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(50, input_shape=(4,)),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['accuracy', 'mean_squared_error'],
        )

    def train(self, x, y):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=3)
        self.model.fit(x, y, verbose=True, epochs=5, batch_size=10, callbacks=[early_stop])
    
    def predict(self, x):
        return np.argmax(self.model.predict(np.array([x])))

