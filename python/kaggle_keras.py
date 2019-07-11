import numpy as np
import pandas as pd
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Model
from keras.layers import Dense, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dropout, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import os
from PIL import Image

# Create a class to store global variables. Easier for adjustments.
class Configuration:
    def __init__(self):
        self.epochs = 4
        self.batch_size = 16
        self.img_width_adjust = 480
        self.img_height_adjust= 360
        self.data_dir = "../data/imgs/train/"

config = Configuration()

# Model Definition
def build_model():
    inputs = Input(shape=(config.img_width_adjust, config.img_height_adjust, 3), name="input")

    # Convolution 1
    conv1 = Conv2D(128, kernel_size=(3, 3), activation="relu", name="conv_1")(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2), name="pool_1")(conv1)

    # Convolution 2
    conv2 = Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv_2")(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2), name="pool_2")(conv2)

    # Convolution 3
    conv3 = Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv_3")(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2), name="pool_3")(conv3)

    # Convolution 4
    conv4 = Conv2D(16, kernel_size=(3, 3), activation="relu", name="conv_4")(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2), name="pool_4")(conv4)

    # Fully Connected Layer
    flatten = Flatten()(pool4)
    fc1 = Dense(1024, activation="relu", name="fc_1")(flatten)

    # output
    output = Dense(10, activation="softmax", name="softmax")(fc1)

    # finalize and compile
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])

    return model

# Setup data, and create split for training, testing 80/20
def setup_data(train_data_dir, val_data_dir, img_width=config.img_width_adjust, img_height=config.img_height_adjust,
               batch_size=config.batch_size):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       validation_split=0.2)  # set validation split

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training')

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation')
    # Note uses training dataflow generator
    return train_generator, validation_generator

def fit_model(model, train_generator, val_generator, batch_size, epochs):
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=val_generator.samples // batch_size,
        verbose=1)  #Verbose: 0: no output, 1: output with status bar, 2: Epochs Only
    return model

# Model Evaluation
def eval_model(model, val_generator, batch_size):
    scores = model.evaluate_generator(val_generator, steps=val_generator.samples // batch_size)
    print("Loss: " + str(scores[0]) + " Accuracy: " + str(scores[1]))

#os.mkdir("./Model")

# Create Data 80/20
train_generator, val_generator = setup_data(config.data_dir, config.data_dir, batch_size=config.batch_size)

# Build the model and show the summary data (note trainable parameters)
model = build_model()
print(model.summary())

# Fit the model
model = fit_model(model, train_generator, val_generator,
                  batch_size=config.batch_size,
                  epochs=config.epochs)

# Save the model
model.save("./Model/distracted_driver.h5")

# Evaluate your model.
eval = eval_model(model, val_generator, batch_size=config.batch_size)
print(eval)