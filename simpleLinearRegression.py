import tensorflow as tf
import numpy as np

# Generate synthetic data
X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
Y = np.array([2, 4, 6, 8, 10], dtype=np.float32)

# Create a simple linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, Y, epochs=500)

# Predict
print(model.predict(np.array([6], dtype=np.float32)))
