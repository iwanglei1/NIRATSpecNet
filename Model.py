from keras import layers
from keras import models
from  keras import metrics
from keras.optimizers import RMSprop
import keras
import someFunction as sF
import tensorflow as tf
import getMilkData as gM
from keras.models import load_model
import datetime
import milkDep as mD

model = load_model('2021-07-15-16-08-04.h5')
model.summary()
# print(tf.__version__)