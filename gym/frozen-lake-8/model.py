import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

class Model:
    def __init__(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, input_shape=(64,)),
            tf.keras.layers.Dense(50),
            tf.keras.layers.Dense(4, activation='relu')
        ])
        self._model.compile(
            loss='mean_squared_error',
            optimizer=tf.train.GradientDescentOptimizer(0.05),
            metrics=['accuracy', 'mean_squared_error'],
        ) # Loss and optimizer

    def predict_one(self, position, render=False):
        prediction = self._model.predict(np.array([np.identity(64)[position]]))
        if render:
            print("Prediction:", position, prediction, "-->", np.argmax(prediction))
        return np.argmax(prediction)

    def normalize_table(self, table):
        def norm(row):
            if np.ptp(row) == 0:
                return np.zeros(row.shape)
            return (row - np.min(row)) / np.ptp(row)
        return np.array([norm(row) for row in table])

    def train(self, table, verbose, epochs=100):
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error', patience=10)
        self._model.fit(
            np.identity(64),
            self.normalize_table(table),
            verbose=verbose,
            epochs=epochs,
            callbacks=[early_stop]
        )
        #if (verbose):
        #self.plot_history(history)