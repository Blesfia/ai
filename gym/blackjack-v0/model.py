import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np

class Model:
    def __init__(self):
        self._model = tf.keras.Sequential([
            tf.keras.layers.Dense(100, input_shape=(3,)),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(2, activation='relu')
        ])
        self._model.compile(
            loss='mean_squared_error',
            optimizer=tf.train.AdamOptimizer(0.1),
            metrics=['accuracy', 'mean_absolute_error', 'mean_squared_error'],
        ) # Loss and optimizer

    def normalize(self, a):
        return (a - np.min(a)) / 31

    def normalize_one(self, state):
        return np.array([(state[0] - 1) / 31, (state[1]-1)/10, (state[2] - 1) / 31])

    def predict_one(self, state, render=False):
        state_to_predict = np.array([self.normalize_one(state)])
        #state_to_predict = np.array([list(state)])
        prediction = self._model.predict(state_to_predict)
        if render:
            print("Prediction:", state, prediction, "-->", np.argmax(prediction))
        return np.argmax(prediction)

    def normalize_table(self, table):
        def norm(column):
            if np.ptp(column) == 0:
                return np.zeros(column.shape)
            return (column - np.min(column)) / np.ptp(column)
        return np.matrix([norm(table[:, i]) for i in range(0,table.shape[1])]).transpose()

    def train(self, x, y, verbose, epochs=100):
        _x = self.normalize_table(x)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_absolute_error', patience=30)
        self._model.fit(
            _x,
            y,
            verbose=verbose,
            epochs=epochs,
            batch_size=100,
            callbacks=[early_stop]
        )
        #if (verbose):
        #self.plot_history(history)