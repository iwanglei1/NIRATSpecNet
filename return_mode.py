from keras import models
from keras import layers

def build_mode():
    model = models.Sequential()
    model.add(layers.Conv1D(64, 5, activation='tanh', input_shape=(700, 1)))
    model.add(layers.MaxPooling1D(3))

    model.add(layers.Conv1D(32, 5, activation='tanh'))
    model.add(layers.MaxPooling1D(3))

    model.add(layers.Conv1D(32, 5, activation='tanh'))
    model.add(layers.GlobalMaxPooling1D())

    model.add(layers.Dense(8))
    model.add(layers.Dense(4))
    model.add(layers.Dense(1))

    model.summary()
    model.compile(optimizer='rmsprop', loss='mse',metrics=['mae'])
    return model