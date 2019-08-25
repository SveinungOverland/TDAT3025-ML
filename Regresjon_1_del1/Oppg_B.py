from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras

print(tf.__version__)

day_length_weight_file = keras.utils.get_file("day_length_weight.csv", "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv")
column_names = ["day", "length", "weight"]

dataset = pd.read_csv(day_length_weight_file, names=column_names, skiprows=1)

print(dataset.tail())

print(dataset.isna().sum())

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

length_weight = dataset[["length", "weight"]]
day = dataset[["day"]]


ax.scatter(length_weight[["length"]], length_weight[["weight"]], day, 'o')
ax.set_xlabel("length")
ax.set_ylabel("weight")
ax.set_zlabel("day")



class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32, [None, 2])
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable(tf.zeros([2, 1]))
        self.b = tf.Variable(tf.zeros([1]))

        # Predictor
        self.f = tf.matmul(self.x, self.W) + self.b

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(self.f - self.y))


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

for epoch in range(40_000):
    session.run(minimize_operation, { model.x: length_weight, model.y: day })
    if epoch % 1000 == 0:
        loss = session.run([model.loss], { model.x: length_weight, model.y: day })
        print(epoch, loss)


W, b, loss = session.run([model.W, model.b, model.loss], { model.x: length_weight, model.y: day })

print("W = %s, b = %s, loss = %s" % (W, b, loss))

day = np.asmatrix(day.to_numpy())
length = np.asmatrix(length_weight[["length"]].to_numpy())
weight = np.asmatrix(length_weight[["weight"]].to_numpy())

print("day min:", day.min())
print("day max:", day.max())
print("length min:", length.min())
print("length max:", length.max())


# print(W)

# x = np.mat([[day.min(), day.max()], [length.min(), length.max()]])
# y = x * W + b

# print(y)

x1_grid, x2_grid = np.meshgrid(np.linspace(length.min(), length.max(), 10), np.linspace(weight.min(), weight.max(), 10))
y_grid = np.empty([10, 10])
for i in range(0, x1_grid.shape[0]):
    for j in range(0, x1_grid.shape[1]):
        y_grid[i, j] = np.mat([[x1_grid[i, j], x2_grid[i, j]]]) * W + b


ax.plot_wireframe(x1_grid, x2_grid, y_grid, color='green')

session.close()

plt.show()






