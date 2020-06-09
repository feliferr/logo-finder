import argparse

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from PIL import Image
import re
import os
from sklearn import preprocessing
from keras.preprocessing.image import ImageDataGenerator
AUTOTUNE = tf.data.experimental.AUTOTUNE


BATCH_SIZE = 128
NUM_EXAMPLES = 110000

def read_lines(file_path):
  names = []
  f = tf.io.gfile.GFile(file_path, mode='r')
  for line in f:
    names.append(line.replace('\n',''))
  return names

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)',text)]

def build_filenames_labels(base_path, relative_paths):
  filenames = []
  labels = []

  for file_path in relative_paths:
    label = file_path.split("/")[-2]
    filename = os.path.join(base_path,file_path)
    filenames.append(filename)
    labels.append(label)
  
  return (filenames, labels)

def _parse_function(filename, label):
  image_string = tf.io.read_file(filename)
  image_decoded = tf.image.decode_jpeg(image_string)
  image_resized = tf.image.resize(image_decoded, [96, 96])
  return image_resized, label

def convert(image, label):
  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
  return image, label

def augment(image,label):
  image,label = convert(image, label)
  image = tf.image.convert_image_dtype(image, tf.float32) # Cast and normalize the image to [0,1]
  image = tf.image.resize_with_crop_or_pad(image, 34, 34) # Add 6 pixels of padding
  image = tf.image.random_crop(image, size=[28, 28, 1]) # Random crop back to 28x28
  image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness

  return image,label

def make_model(num_classes):
  model = tf.keras.Sequential()
  model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.MaxPooling2D((2, 2)))
  model.add(layers.Conv2D(64, (3, 3), activation='relu'))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(num_classes-1))
  model.compile(optimizer = 'adam',
                loss=tf.losses.CategoricalCrossentropy(),
                metrics=['accuracy'])
  return model

def train_model(train_base_path, 
                test_base_path, 
                train_image_meta, 
                test_image_meta, 
                classes_meta,
                job_dir='./tmp/logo_classification', **args):

  # classe_names = read_lines(classes_meta)
  # train_image_paths = read_lines(train_image_meta)
  # test_image_paths = read_lines(test_image_meta)

  # train_image_paths.sort(key=natural_keys)
  # test_image_paths.sort(key=natural_keys)

  # num_train_examples = len(train_image_paths)
  # num_classes = len(classe_names)
  # num_test_examples = len(test_image_paths)

  # train_filenames, train_labels = build_filenames_labels(train_base_path, train_image_paths)
  # test_filenames, test_labels = build_filenames_labels(test_base_path, test_image_paths)

  # lb = preprocessing.LabelBinarizer()
  # lb.fit(train_labels)

  # train_onehot_labels = lb.transform(train_labels)
  # test_onehot_labels = lb.transform(test_labels)

  # tfc_train_files = tf.constant(train_filenames)
  # tfc_train_labels = tf.constant(train_onehot_labels)
  # tfc_test_files = tf.constant(test_filenames)
  # tfc_test_labels = tf.constant(test_onehot_labels)

  # dataset_train = tf.data.Dataset.from_tensor_slices((tfc_train_files, tfc_train_labels))
  # dataset_train = dataset_train.map(_parse_function)
  # dataset_test = tf.data.Dataset.from_tensor_slices((tfc_test_files, tfc_test_labels))
  # dataset_test = dataset_test.map(_parse_function)

  # non_augmented_train_batches = (
  #   dataset_train
  #   # Only train on a subset, so you can quickly see the effect.
  #   #.take(NUM_EXAMPLES)
  #   #.cache()
  #   .shuffle(num_train_examples//4)
  #   # No augmentation.
  #   .map(convert, num_parallel_calls=AUTOTUNE)
  #   .batch(BATCH_SIZE)
  #   .prefetch(AUTOTUNE)
  # )

  # validation_batches = (
  #   dataset_test
  #   #.take(1024)
  #   #.cache()
  #   .map(convert, num_parallel_calls=AUTOTUNE)
  #   .batch(2*BATCH_SIZE)
  # )

  # model = make_model(num_classes)

  # history = model.fit(non_augmented_train_batches, epochs=50, validation_data=validation_batches)

  os.system("gsutil -m cp 'gs://rec-alg/datasets/logo-2k/train_and_test.zip' .")
  unzip_result = os.system("unzip train_and_test.zip")
  print("unzip ran with exit code %d" % unzip_result)
  mkdir_result1 = os.system("mkdir train_ & cp -r 'train_and_test/train/*/*' train_")
  print("mkdir_result train_ ran with exit code %d" % mkdir_result1)
  mkdir_result2 = os.system("mkdir test_ & cp -r 'train_and_test/test/*/*' test_")
  print("mkdir_result test_ ran with exit code %d" % mkdir_result2)

  # create generator
  datagen = ImageDataGenerator()
  # prepare an iterators for each dataset
  train_it = datagen.flow_from_directory('train_',
                                        batch_size=128, 
                                        target_size=(96, 96),
                                        shuffle=True,
                                        color_mode="rgb",
                                        class_mode='categorical')

  #val_it = datagen.flow_from_directory('data/validation/', class_mode='binary')
  test_it = datagen.flow_from_directory('test_', 
                                        batch_size=128,
                                        target_size=(96, 96),
                                        shuffle=True,
                                        color_mode="rgb",
                                        class_mode='categorical')

  base_model = tf.keras.applications.InceptionV3(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(96, 96, 3),
    include_top=False)
  base_model.trainable = False

  inputs = tf.keras.Input(shape=(96, 96, 3))
  # We make sure that the base_model is running in inference mode here,
  # by passing `training=False`. This is important for fine-tuning, as you will
  # learn in a few paragraphs.
  x = base_model(inputs, training=False)
  # Convert features of shape `base_model.output_shape[1:]` to vectors
  x = tf.keras.layers.GlobalAveragePooling2D()(x)
  # x = tf.keras.layers.Flatten()(x)
  # x = tf.keras.layers.Dense(2048, activation='relu')(x)
  outputs = tf.keras.layers.Dense(2340, activation='softmax')(x)
  custom_model = tf.keras.Model(inputs, outputs)
  custom_model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  custom_model.summary()

  history = custom_model.fit_generator(
      train_it,
      epochs=100,
      validation_data=test_it
  )


if __name__ == '__main__':
    # Parse the input arguments for common Cloud ML Engine options
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--train-base-path',
      help='Cloud Storage bucket or local base path to training data')
    parser.add_argument(
      '--test-base-path',
      help='Cloud Storage bucket or local base path to test data')
    parser.add_argument(
      '--train-image-meta',
      help='Cloud Storage bucket or local path to train image metadata')
    parser.add_argument(
      '--test-image-meta',
      help='Cloud Storage bucket or local path to test image metadata')
    parser.add_argument(
      '--classes-meta',
      help='Cloud Storage bucket or local path to classes metadata')
    parser.add_argument(
      '--job-dir',
      help='Cloud storage bucket to export the model and store temp files')
    args = parser.parse_args()
    arguments = args.__dict__
    train_model(**arguments)