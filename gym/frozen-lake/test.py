
import tensorflow as tf
import numpy as np
model = tf.keras.Sequential([
    tf.keras.layers.Dense(50, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(4)
])
model.compile(
    loss='mean_squared_error',
    optimizer=tf.train.GradientDescentOptimizer(0.05),
    metrics=['accuracy', 'mean_absolute_error', 'mean_squared_error']) # Loss and optimizer

# np.identity(16)[0]
table = np.zeros((1,4))
data = np.array([[1, 2, 3]])
print("Data", data.shape)
prediction = model.predict(data)
print("Prediction:", prediction)
model.fit(np.array([[1, 2, 3]]), table)

model.train_function