import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt
import tqdm.auto
import photos


tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.WARN)
tqdm.tqdm = tqdm.auto.tqdm


# print(tf.__version__)


# Directories
folder = 'images'
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder)
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Number of pictures
total_train = 0
for class_path in os.listdir(train_dir):
    total_train += len([item for item in os.listdir(os.path.join(train_dir, class_path))])
total_val = 0
for class_path in os.listdir(validation_dir):
    total_val += len([item for item in os.listdir(os.path.join(validation_dir, class_path))])

EPOCHS = 14
BATCH_SIZE = 8  # Number of training examples to process before updating our models variables
IMG_SHAPE = 150  # Our training data consists of images with width of 150 pixels and height of 150 pixels

# Get input data
train_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_SHAPE, IMG_SHAPE),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                              directory=validation_dir,
                                                              shuffle=True,
                                                              target_size=(IMG_SHAPE, IMG_SHAPE),
                                                              class_mode='binary')
dict_labels = val_data_gen.class_indices


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plot_images(data_gen, num_data):
    images_arr, _ = data_gen

    fig, axes = plt.subplots(1, 2, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr[:num_data], axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()


# plot_images(next(train_data_gen), 2)  # Plot images 0-1

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SHAPE, IMG_SHAPE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(30, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# model.summary()


history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(total_train / float(BATCH_SIZE))),
    epochs=EPOCHS,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(total_val / float(BATCH_SIZE)))
)


def plot_training_data(history, EPOCHS):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(EPOCHS)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.savefig('./foo.png')
    plt.show()


plot_training_data(history, EPOCHS)
prediction_dir = older_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')


def test_with_camera():
    prediction_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data
    prediction_data_gen = prediction_image_generator.flow_from_directory(directory=prediction_dir,
                                                                         batch_size=1,
                                                                         shuffle=False,
                                                                         target_size=(IMG_SHAPE, IMG_SHAPE),
                                                                         class_mode='binary')

    predictions = model.predict_generator(prediction_data_gen)

    for prediction in predictions:
        label_prediction = [k for k, v in dict_labels.items() if v == np.argmax(prediction)][0]
        print(f'Prediccion: {label_prediction}')


photos.TestPhoto(test_with_camera)


# # Funcionaaaaaaaaa :)
# prediction_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data
# # prediction_data_gen = prediction_image_generator.flow_from_directory(batch_size=BATCH_SIZE, directory=prediction_dir, shuffle=True, target_size=(IMG_SHAPE, IMG_SHAPE), class_mode='binary')
# prediction_data_gen = prediction_image_generator.flow_from_directory(directory=prediction_dir,
#                                                                      batch_size=1,
#                                                                      shuffle=False,
#                                                                      target_size=(IMG_SHAPE, IMG_SHAPE),
#                                                                      class_mode='binary')
#
# predictions = model.predict_generator(prediction_data_gen)
# print(f'Prediccion0: {predictions}')
#
# for prediction in predictions:
#     print(f'Prediccion1: {np.argmax(prediction)}')






