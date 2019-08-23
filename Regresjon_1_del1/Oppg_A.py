import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

print(tf.__version__)

length_weight_file = keras.utils.get_file("length_weight.csv", "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv")
column_names = ["length", "weight"]
raw_dataset = pd.read_csv(length_weight_file, names=column_names, skiprows=1)


dataset = raw_dataset.copy()

print(dataset.tail())

print(dataset.isna().sum())

fig, ax = plt.subplots()


ax.plot(dataset[["length"]], dataset[["weight"]], 'o')
ax.set_xlabel("length")
ax.set_ylabel("weight")


print(dataset[["length"]].tail())

class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # Predictor
        f = tf.matmul(self.x, self.W) + self.b

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(f - self.y))

class LinearRegressionVisualizer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def f(self, x):
        return x * self.W + self.b

    # Mean Squared Error
    def loss(self, x, y):
        return np.mean(np.square(self.f(x) - y))



model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.00015132).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

length = np.asmatrix(dataset[["length"]].to_numpy())
weight = np.asmatrix(dataset[["weight"]].to_numpy())

for epoch in range(360000):
    session.run(minimize_operation, { model.x: length, model.y: weight })
    if epoch % 1000 == 0:
        loss = session.run([model.loss], { model.x: length, model.y: weight })
        print(epoch, loss)


# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], { model.x: length, model.y: weight })
print("W = %s, b = %s, loss = %s" % (W, b, loss))

session.close()

visualizer = LinearRegressionVisualizer(W, b)
print("Predicted loss:", visualizer.loss(length, weight))

x = np.mat([[length.min()], [length.max()]])

ax.plot(x, visualizer.f(x), label='$y = f(x) = xW+b$')

ax.legend()
plt.show()
