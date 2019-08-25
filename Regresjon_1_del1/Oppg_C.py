import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

day_circumference_file = keras.utils.get_file("day_circumference.csv", "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv")
column_names = ["day", "circumference"]

dataset = pd.read_csv(day_circumference_file, names=column_names, skiprows=1)

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

print(dataset)

print(dataset.isna().sum())




def build_model():
    model = keras.Sequential([
        layers.Dense(10, activation=tf.nn.relu, input_shape=[1]),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.relu),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.00001)
    model.compile(loss="mean_squared_error",
                  optimizer=optimizer,
                  metrics=["mean_absolute_error", "mean_squared_error"])
    return model

model = build_model()

print(model.summary())

class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 1000 == 0: print(epoch, logs['loss'])
        

EPOCHS = 6000

history = model.fit(dataset[["day"]],
                    dataset[["circumference"]],
                    epochs=EPOCHS,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[PrintDot()])

hist = pd.DataFrame(history.history)

hist["epoch"] = history.epoch
print()
print(hist.tail())

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.show()

plot_history(history)

fig, ax = plt.subplots()
ax.plot(dataset[["day"]], dataset[["circumference"]], 'o')
ax.set_xlabel("day")
ax.set_ylabel("circumference")

x = np.linspace(dataset[["day"]].min(), dataset[["day"]].max(), 100)
y = model.predict(x).flatten()

ax.plot(x, y)


plt.show()
