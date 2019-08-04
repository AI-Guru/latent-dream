from tensorflow.keras import models, layers, callbacks
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


z_size = 64
batch_size = 100
experience_length = 100


def generator(batch_size):
    dataset_path = "rnn_dataset-200.p"
    experiences = pickle.load(open(dataset_path, "rb"))
    while True:

        x_inputs  = []
        y_outputs = []
        while len(x_inputs) != batch_size:
            experience = random.choice(experiences)
            if len(experience) < experience_length:
                continue
            random_index = random.randint(0, len(experience) - experience_length - 1 - 1)
            sub_experience = experience[random_index:random_index + experience_length + 1]
            observations = np.array([s[0] for s in sub_experience])
            #print(observations.shape)
            x_inputs.append(observations[0:experience_length])
            y_outputs.append(observations[-1])

        x_inputs  = np.array(x_inputs)
        y_outputs = np.array(y_outputs)
        #print(x_inputs.shape, y_outputs.shape)
        yield x_inputs, y_outputs

#model = models.Sequential()
#model.add(layers.LSTM(z_size, input_shape=(experience_length, z_size), activation="linear"))
#model.summary()


model = models.Sequential()
model.add(layers.GRU(512, input_shape=(experience_length, z_size), activation="relu"))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dense(z_size, activation="linear"))
model.summary()

model.compile(
    optimizer="rmsprop",
    loss="mse",
    metrics=["mae"]
)

datetime_string = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir="logs/rnn-" + datetime_string
tensorboard_callback = callbacks.TensorBoard(log_dir=logdir)

model.fit_generator(
    generator(batch_size),
    steps_per_epoch=100,
    epochs=1000,
    callbacks=[tensorboard_callback]
)

model.save("rnn-{}.h5".format(datetime_string))
