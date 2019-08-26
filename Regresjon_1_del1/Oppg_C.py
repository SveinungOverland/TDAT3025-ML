import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
# from tensorflow.keras import backend as K

print(tf.__version__)

day_circumference_file = keras.utils.get_file("day_circumference.csv", "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv")
column_names = ["day", "circumference"]

dataset = pd.read_csv(day_circumference_file, names=column_names, skiprows=1)

print(dataset)

print(dataset.isna().sum())


class LinearRegressionModel:
    def __init__(self):
        # Model input
        self.x = tf.placeholder(tf.float32)
        self.y = tf.placeholder(tf.float32)

        # Model variables
        self.W = tf.Variable([[0.0]])
        self.b = tf.Variable([[0.0]])

        # # Predictor
        # self.f = 20 * tf.nn.sigmoid(tf.matmul(self.x, self.W) + self.b) + 31 

        # Mean Squared Error
        self.loss = tf.reduce_mean(tf.square(self.f(self.x) - self.y))

    def f(self, x):
        return 20 * tf.sigmoid(tf.matmul(x, self.W) + self.b) + 31


class Visualizer:
    def __init__(self, W, b):
        self.W = W
        self.b = b

    # Predictor
    def predict(self, x):
        return 20 * (1 / (1 + np.exp(-(x * np.mat(self.W) + np.mat(self.b))))) + 31


model = LinearRegressionModel()

# Training: adjust the model so that its loss is minimized
minimize_operation = tf.train.GradientDescentOptimizer(0.0000000001).minimize(model.loss)

# Create session object for running TensorFlow operations
session = tf.Session()

# Initialize tf.Variable objects
session.run(tf.global_variables_initializer())

day = np.asmatrix(dataset[["day"]].to_numpy())
circumference = np.asmatrix(dataset[["circumference"]].to_numpy())

for epoch in range(6_000):
    session.run(minimize_operation, { model.x: day, model.y: circumference })
    if epoch % 1000 == 0:
        loss = session.run([model.loss], { model.x: day, model.y: circumference })
        print(epoch, loss)


# Evaluate training accuracy
W, b, loss = session.run([model.W, model.b, model.loss], { model.x: day, model.y: circumference })
print("W = %s, b = %s, loss = %s" % (W, b, loss))


session.close()


x = np.linspace(day.min(), day.max(), 100).reshape(-1, 1)
y = Visualizer(np.mat(W), np.mat(b)).predict(x)

fig, ax = plt.subplots()

ax.plot(day, circumference, 'o')
ax.set_xlabel("day")
ax.set_ylabel("circumference")

ax.plot(x, y)

ax.legend()
plt.show()
