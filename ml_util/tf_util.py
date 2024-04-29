"""
TensorFlow Utility Functions
Author: Tony Held (tony.held@arb.ca.gov)
"""
###############################################################################
# ML Related Imports
###############################################################################

import IPython.display
import collections
import copy
import importlib
import itertools
import json
import os
import pathlib
import random
import shutil
import sys
import tarfile
import urllib.request
import zipfile

from datetime import datetime
from os.path import exists
from packaging import version
from pathlib import Path
from urllib.parse import urlparse
from zipfile import ZipFile

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import tensorflow as tf

from keras import Sequential, layers, models, regularizers
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory, img_to_array, load_img, plot_model
from sklearn import set_config
from sklearn.compose import ColumnTransformer, make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from tensorflow import keras

# tensorflow_hub may have compatibility issues, it may be best to have it be optionally loaded
try:
  import tensorflow_hub as hub
except ModuleNotFoundError:
  print(f"tensorflow_hub package not available")


###############################################################################
# Downloading modeling data from urls
###############################################################################

def url_to_local_dir(url_str,
                     target_dir='.',
                     decompress=True,
                     ):
  """
  Download a file from url to a local directory with the option to unzip/decompress.
  If the file is already present on the local drive, it will not be downloaded again.

  Args:
    url_str (str):
      url of the remote file
    target_dir (str|Path):
      location of local directory to save url file contents
    decompress (bool):
      True to decompress file if it s a known type
  Returns:
    full_file_name: full path for potential file download
  Notes:
    * If you are on system that uses Google Drive that backs up to the network,
      you will likely want the target_dir to be on a non-backed up directory to avoid
      the backing up of files that should not be on gdrive.
  Examples:
    * tfu.url_to_local_dir("https://github.com/ageron/data/raw/main/housing.tgz",
      "/Users/tony/ml/temp/housing")
    * tfu.url_to_local_dir("https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip")
  """
  # Ensure target directory is a Path
  if not isinstance(target_dir, Path):
    target_dir = Path(target_dir)

  # Create the target directory if needed
  Path(target_dir).mkdir(parents=True, exist_ok=True)

  # find full path for potential file download and file suffix
  full_file_name = file_info(url_str, target_dir)

  if full_file_name.is_file():
    print(f"File already exists, no download required.")
    print(f"\t{full_file_name}")
  else:
    print(f"Downloading/Creating File: {full_file_name}")
    urllib.request.urlretrieve(url_str, full_file_name)
    if decompress is True:
      decompress_file(full_file_name, target_dir)
  return full_file_name


def file_info(url_str, target_dir, verbose=0):
  """
  Given url of a file of interest, and a local directory to download to,
  return the full path of the downloaded file on the local system.

  Args:
    url_str (str):
      url of the remote file
    target_dir (str|Path):
      location of local directory to save url file contents
    verbose (int):
      0 to suppress diagnostics

  Returns:
    full_file_name: full path for potential file download
  """
  # Parse url to find filename and suffix
  url_parsed = urlparse(url_str)
  url_path = Path(url_parsed.path)
  url_file_name = url_path.name
  url_suffix = url_path.suffix
  full_file_name = target_dir / url_file_name
  if verbose != 0:
    print(f"URL of requested file: {url_str}")
    print(f"Filename: {url_file_name}")
    print(f"File Type: {url_suffix}")
    print(f"Target directory for download: {target_dir}")
    print(f"Full path of file on local system: {full_file_name}")
  return full_file_name


def decompress_file(full_file_name, target_dir):
  """
  Decompress a file if possible.

  Args:
    full_file_name (Path):
      full path to compressed file
    target_dir (str|Path):
      compressed file's local directory
  """
  print(f"Attempting to decompress file: {full_file_name}")
  file_suffix = full_file_name.suffix

  if file_suffix in ['.tgz']:
    with tarfile.open(full_file_name) as compressed_file:
      compressed_file.extractall(path=target_dir)
  elif file_suffix in ['.zip']:
    with ZipFile(full_file_name, 'r') as zObject:
      zObject.extractall(path=target_dir)
  else:
    print(f"{file_suffix} is not a known compressed file type.  No compression attempted.")

  print(f"target_dir contents:")
  print(os.listdir(target_dir))


###############################################################################
# File viewing & management
###############################################################################
def mk_dir(dir_name):
  """
  Make a directory recursively creating directory structure as needed.

  Args:
    dir_name (str|Path): Directory to create.
      Parent directories will be created if needed.
  """
  if not exists(dir_name):
    print(f'Creating/Confirming directory: {dir_name}')
    Path(dir_name).mkdir(parents=True, exist_ok=True)


def walk_directory(directory):
  """
  Walk through a directory and its subdirectories to determine file counts
  Args:
      directory (str|Path):
  """
  for dirpath, dirnames, filenames in os.walk(directory):
    print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")


def directories_and_files(base_directory):
  """
  Return the full paths to directories and files in a base directories.

  Will recursively search subdirectories.

  Args:
    base_directory (str|Path):

  Returns:
    (directories, files)
      directories: list of paths of directories starting with target_dir
      files: list of files paths in target_dir and its subdirectories
  """
  directory_list = []
  file_list = []
  for directory, sub_directories, files in os.walk(base_directory):
    print(f"There are {len(sub_directories)} directories and {len(files)} files in '{directory}'.")
    dir_path = Path(directory)
    directory_list.append(dir_path)
    for file in files:
      file_list.append(dir_path / file)
  print(f"A total of {len(directory_list)} directories and {len(file_list)} files detected.")
  return directory_list, file_list


def class_names(base_directory):
  """
  Given a base directory that has a sub folder named 'train', find the associated class label names
  with the training subdirectory.

  Args:
    base_directory (str|Path):

  Returns (list):
    sorted list of subdirectories (assumed to be class names) for an image directory structure
  """
  base_directory = Path(base_directory)
  train_directory = base_directory / 'train'
  _, sub_directories, _ = next(os.walk(train_directory))
  sub_directories.sort()
  print(f"A total of {len(sub_directories)} sub directories (classes) were found in the training directory")
  return sub_directories


def dir_class_names(directory):
  """
  Find the classification names associated with an image training directory.
  Args:
    directory (str|Path):
      Directory with subdirectories named after image classification types
  Returns:
    Array of classification names
  """
  directory = pathlib.Path(directory)
  if directory.exists() and directory.is_dir():
    class_names = np.array(sorted([item.name for item in directory.glob('*')]))
    # print(f'The class names are: {class_names}\n')
  else:
    class_names = None
    print(f'{directory} does not appear to be a directory')
  return class_names


def random_sample_directory(directory, number_files=1):
  """
  Randomly sample files from a directory.

  Args:
    directory (str|Path): directory to sample files from
    number_files (int): number of files to return
  Returns:
    a list of path names sampled from directory
  """
  directory = pathlib.Path(directory)
  file_names = random.sample(os.listdir(directory), number_files)
  files = [directory.joinpath(file_name) for file_name in file_names]
  return files


def random_sample_single_class(directory, class_name, number_files=1):
  """
  Randomly sample files from a directory associated with an image class name.

  Args:
    directory (str|Path): directory to sample files from
    number_files (int): number of files to return
    class_name (str): image class of interest
  Returns:
    a list of path names sampled from directory with stated image class name
  """
  directory = pathlib.Path(directory)
  class_directory = directory.joinpath(class_name)
  file_names = random_sample_directory(class_directory, number_files)
  return file_names


def random_sample_all_classes(directory, number_files):
  """
  Randomly sample files from a directory for each image class type.


  Args:
    directory (str|Path): directory to sample files from
    number_files (int): number of files to return for each class type

  Returns:
    (file_names, class_names)
      file_names: list of path names sampled from directory
      file_class: list of label class associated with file_names
  """
  directory = pathlib.Path(directory)
  class_names = dir_class_names(directory)
  file_names = []
  file_class = []
  for class_name in class_names:
    files = random_sample_single_class(directory, class_name, number_files)
    file_names.extend(files)
    file_class.extend([class_name] * number_files)
  return file_names, file_class


def split_directory(base_dir,
                    old_dir="train",
                    new_dir="val",
                    fraction=0.2,
                    categories=None):
  """
  Move a fraction of files from an existing directory's subdirectories to a new directory.
  Helpful to create a valuation set from training or testing data.
  Args:
      base_dir: root of dataset file system
      old_dir: existing directory to move samples from
      new_dir: target directory to move samples to
      fraction: (0 to 1) fraction of files to move
      categories: subdirectories of the old_dir
  Returns:
  """
  if categories is None:
    categories = ['neg', 'pos']
  base_dir = pathlib.Path(base_dir)
  old_dir = base_dir / old_dir
  new_dir = base_dir / new_dir

  print(f"Directory structure before splitting")
  walk_directory(base_dir)

  for category in categories:
    os.makedirs(new_dir / category)
    files = os.listdir(old_dir / category)
    random.Random(1337).shuffle(files)
    num_new_samples = int(fraction * len(files))
    new_files = files[-num_new_samples:]
    print(f"Moving {num_new_samples} from {old_dir / category} to {new_dir / category}")
    for fname in new_files:
      shutil.move(old_dir / category / fname,
                  new_dir / category / fname)

  print(f"Directory structure after splitting")
  walk_directory(base_dir)


def read_embeddings_file(file_path):
  """
  Read in an embedding file that is space delimited with the format
      word x1, x2, x3, ...
  Returns:
      a dictionary.  the key is individual words and the values
      are n-d numpy array associated with the embedding
  """
  print(f"Processing file: {file_path}")

  embeddings = {}
  with open(file_path) as f:
    for line in f:
      # maxsplit=1 will break the line into 2 parts, the word and then the coefs
      word, coefs_text = line.split(maxsplit=1)
      coefs = np.fromstring(coefs_text, "f", sep=" ")
      embeddings[word] = coefs
      # print(f"line: {line}\n word: {word}\n coefs_text: {coefs_text}")
      # print(f"coefs (length={len(coefs)})\n{coefs}")
      # break

  num_dim = 0
  for key, value in embeddings.items():
    num_dim = len(value)
    break
  print(f"Found {len(embeddings)} word vectors with embedding dimension of {num_dim}.")
  return embeddings


def get_embedding_matrix(max_tokens,
                         embedding_dim,
                         text_vectorization,
                         embedding_dict):
  """
  Create an embedding matrix suitable for layers.Embedding embeddings_initializer
  argument from an adapted TextVectorization and pre-trained embedding dict.
  Args:
    max_tokens: Maximum length (in words) of an input string
    embedding_dim: number of dimensions in the pre-trained embedding
    text_vectorization: TextVectorization obj with output_mode="int"
    embedding_dict: pre-trained embedding created with tfu.read_embeddings_file
  Returns:
    numpy array with shape (max_tokens, embedding_dim) suitable for
    embeddings_initializer argument of layers.Embedding
  Notes:
    1) The matrix will have the same row indexing as the text_vectorization.get_vocabulary
    array truncated to the first max_tokens rows
    2) If a word is not in the embedding_dict, it will be represented as a zero vector
    3) Standford's Glove embedding does not have '[UNK]', so that will be a zero vector
    4) methodology is based on chollet's implementation, an alternative implementation would be
        for i, word in enumerate(vocab_array[:max_tokens]):
          embedding_vector = embedding_dict.get(word)
          if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
  """
  # word embedding defaults to a zero vector
  # todo - maybe specify the dtype? Default is numpy.float64
  embedding_matrix = np.zeros((max_tokens, embedding_dim))

  vocab_array = text_vectorization.get_vocabulary()  # given an index, find the word
  vocab_dict = dict(zip(vocab_array, range(len(vocab_array))))  # given a word, find the index

  for word, i in vocab_dict.items():
    if i < max_tokens:
      embedding_vector = embedding_dict.get(word)
      if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
  return embedding_matrix


###############################################################################
# Splitting Data
###############################################################################

def split_data(X, y, test_size, val_size=0, random_state=42, verbose=1):
  """
  Split dataset into training, testing, and optional cross-validation
  datasets using sklearn.model_selection.train_test_split.

  Args:
    X: Features
    y: Labels
    test_size (float): fraction of data for testing (<0.0 and <1.0)
    val_size (float): fraction of data for cross-validation (<=0.0 and <1.0)
    random_state (int): seed for shuffling
    verbose: =0 to suppress diagnostics

  Returns:
    (X_train, X_test, X_val, y_train, y_test, y_val)

  See Also:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
  """
  assert (test_size > 0) and (test_size < 1)
  assert (val_size >= 0) and (val_size < 1)
  assert test_size + val_size < 1

  # Fraction of data to hold back from training
  # Round to 5 significant digits to avoid machine rounding issues
  hold_back = round(test_size + val_size, 5)
  train_size = 1 - hold_back

  # Split the data into training and holdback subsets
  X_train, X_hold, y_train, y_hold = train_test_split(X,
                                                      y,
                                                      test_size=hold_back,
                                                      random_state=random_state)

  # Split the holdback into test and cross validation
  # Note: they are already shuffled, so you don't need to reshuffle
  test_fraction_of_holdback = test_size / hold_back
  test_cutpoint = int(round(test_fraction_of_holdback * len(y_hold)))

  X_test = X_hold[:test_cutpoint]
  y_test = y_hold[:test_cutpoint]
  X_val = X_hold[test_cutpoint:]
  y_val = y_hold[test_cutpoint:]

  # Print diagnostics if requested
  if verbose > 0:
    # Requested sizes
    request_train = train_size * 100
    request_test = test_size * 100
    request_val = val_size * 100
    request_hold = request_test + request_val

    # Actual sizes
    percent_train = len(X_train) / len(X) * 100
    percent_hold = len(X_hold) / len(X) * 100
    percent_test = len(X_test) / len(X) * 100
    percent_val = len(X_val) / len(X) * 100

    print(
      f'Requested data splits: '
      f'{request_train:05.2f}% training, '
      f'{request_hold:05.2f}% holdback, '
      f'{request_test:05.2f}% testing, '
      f'{request_val:05.2f}% cross-val')
    print(
      f'Actual    data splits: '
      f'{percent_train:05.2f}% training, '
      f'{percent_hold:05.2f}% holdback, '
      f'{percent_test:05.2f}% testing, '
      f'{percent_val:05.2f}% cross-val')

    print(f'Shapes of input/output data')
    print(f'X      : {X.shape}\t y      : {y.shape}')
    print(f'X_train: {X_train.shape}\t y_train: {y_train.shape}')
    print(f'X_hold : {X_hold.shape}\t y_hold : {y_hold.shape}')
    print(f'X_test : {X_test.shape}\t y_test : {y_test.shape}')
    print(f'X_val  : {X_val.shape}\t y_val  : {y_val.shape}')

  return X_train, X_test, X_val, y_train, y_test, y_val


def split_data_0(X, y, fraction_validation=0.3):
  """
  Split data into training and validation without using sklearn.

  Args:
    X (ArrayLike):
    y (ArrayLike):
    fraction_validation (float):

  Returns:
    (X_train, y_train, X_val, y_val)

  Notes:
    * Depreciated, use split_data instead.
  """
  indices_permutation = np.random.permutation(len(X))
  X_shuffled = X[indices_permutation]
  y_shuffled = y[indices_permutation]

  num_validation_samples = int(fraction_validation * len(X))
  X_val = X_shuffled[:num_validation_samples]
  y_val = y_shuffled[:num_validation_samples]
  X_train = X_shuffled[num_validation_samples:]
  y_train = y_shuffled[num_validation_samples:]

  return X_train, y_train, X_val, y_val


###############################################################################
# Image Processing
###############################################################################
def image_files_to_tensor(file_names, img_shape, channels=3, rescale=255.):
  """
  Reads images from file_names and returns an image tensor.

  Args:
    file_names (list[str|Path]): file names with images
    img_shape (int): size in pixels (for both height and width)
    channels (int): Channel determines the color coding type
    rescale (float):
      Divide image value by rescale to result in a normalized vector

  Notes:
    * Channels,   Output Type
    * 0,          The original number of channels in the image.
    * 1,          Outputs the images as greyscale.
    * 3,          Outputs the images as RGB.
    * 4,          Outputs image as RGBA

  Returns:
    image tensor created from file_names
  """
  # List of files as decoded images
  images = []
  for file_name in file_names:
    image = tf.io.read_file(str(file_name))

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    image = tf.image.decode_image(image, channels=channels)

    # Resize the image (to the same size our model was trained on)
    image = tf.image.resize(image, size=[img_shape, img_shape])
    images.append(image)

  # Convert list of tensors to single tensor
  images = tf.stack(images, axis=0)

  if rescale:
    # Rescale the image (get all values between 0 and 1)
    images = images / rescale
  return images


def generator_labels(directory, class_mode='binary'):
  """
  Return the y labels (as int) associated with an image directory.

  Args:
    directory: Base location of image files
    class_mode (str):
      Can be 'binary' or 'categorical' depending on type of classification problem

  Returns:
    (y_labels, counts)
      y_labels - list of class indices (int) associated with images
      counts - dictionary of class indices and frequency counts

  Notes:
    1) Shuffle is set to False in order to preserve order with predictions.
    2) Uses ImageDataGenerator flow_from_directory function for use in predicted/actual analyses.

  """
  # Note, the generator will loop endlessly, so an iteration break is required
  # The last batch may be smaller than the requested batch size
  # if the length of the dataset is not perfectly divisible by batch_size
  image_processor = ImageDataGenerator()
  image_generator = image_processor.flow_from_directory(directory,
                                                        batch_size=100,
                                                        class_mode=class_mode,
                                                        shuffle=False)
  num_batches = len(image_generator)
  # print(f"num_batches: {num_batches}")
  y_labels = []
  for i, (x, y) in enumerate(image_generator):
    if i == num_batches:
      break
    y_labels.extend(y)

  y_labels = np.array(y_labels, dtype='int')

  # if the class_mode is 'categorical', the y_labels are one-hot encoded
  if class_mode == 'categorical':
    y_labels = y_labels.argmax(axis=1)

  counts = collections.Counter(y_labels)
  print(f'Label counts for testing data:\n\t{counts}')

  return y_labels, counts


###############################################################################
# Plotting
###############################################################################

def plot_model_summary(model):
  """
  Plot kera model layer information.

  Example Usage:
    import IPython.display
    fig = plot_model_summary(model_1)
    IPython.display.display(fig)
  """
  model.summary()
  fig = plot_model(model, show_shapes=True)
  return fig


def plot_history_metric(history,
                        metric='mae',
                        title=None):
  """
  Plot single metric as a function of epochs
  for training data and optionally for cross-validation data.

  Args:
    history (tf.keras.callbacks.History | pd.core.frame.DataFrame):
      history obj can of type:
        1) be returned from model fitting
        2) pandas (model history saved as a csv and imported as a DataFrame)
    metric (str):
      metric to visualize
    title (str|None):
      figure title
  """
  # Extract history dictionary from callback or dataframe if necessary
  if isinstance(history, tf.keras.callbacks.History):
    history = history.history
  elif isinstance(history, pd.core.frame.DataFrame):
    history = history.to_dict('list')

  fig = plt.figure()
  plt.plot(history[metric], 'b', label=metric)
  val_metric = f'val_{metric}'
  if val_metric in history:
    plt.plot(history[val_metric], 'gx', label=val_metric)
  plt.xlabel('Epoch')
  plt.legend()
  if title:
    plt.title(title)
  plt.show()
  return fig


def plot_all_history_metrics(history, title=None):
  """
  Plot all history metrics as a function of epochs

  Args:
    history (tf.keras.callbacks.History | pd.core.frame.DataFrame):
      history obj can of type:
        1) be returned from model fitting
        2) pandas (model history saved as a csv and imported as a DataFrame)
    title (str|None):
      figure title

  """
  # Extract history as dict and convert to dataframe if necessary
  if isinstance(history, tf.keras.callbacks.History):
    history = history.history
  if not isinstance(history, pd.core.frame.DataFrame):
    history = pd.DataFrame(history)

  history.plot()
  plt.ylabel("metrics")
  plt.xlabel("epochs")
  if title:
    plt.title(title)
  plt.show()


def plot_loss_and_accuracy(history, title=""):
  """
  Plots loss and accuracy for training and validation data as separate figures.

  Args:
    history:
      history obj can of type:
        1) be returned from model fitting
        2) pandas (model history saved as a csv and imported as a DataFrame)
        3) dictionary
    title (str): Optional caption to include with figure title
  """
  # Extract history dictionary from callback or dataframe if necessary
  if isinstance(history, tf.keras.callbacks.History):
    history = history.history
  elif isinstance(history, pd.core.frame.DataFrame):
    history = history.to_dict('list')

  loss = history['loss']
  val_loss = history['val_loss']

  accuracy = history['accuracy']
  val_accuracy = history['val_accuracy']

  epochs = range(len(history['loss']))

  # Plot loss
  plt.figure()
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss: ' + title)
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy: ' + title)
  plt.xlabel('Epochs')
  plt.legend()


def plot_history_list(history_list,
                      metric="mae",
                      figsize=(8, 8),
                      ):
  """
  Plot a model history and training metric for multiple sequential histories created by
  refitting a model.

  Args:
    history_list: list-like sequential histories from refitting
    metric: metric to plot ('mae', 'accuracy', etc)
    figsize:

  Returns: figure

  """
  loss_train = []
  loss_val = []
  metric_train = []
  metric_val = []
  end_epochs = []
  start_epoch = 0

  for history in history_list:
    # Extract loss and metric of interest from training and testing
    loss_train.extend(history.history["loss"])
    loss_val.extend(history.history["val_loss"])
    metric_train.extend(history.history[metric])
    metric_val.extend(history.history["val_" + metric])
    # Find number of epochs in each history
    num_epochs = len(history.history["loss"])
    end_epoch = start_epoch + num_epochs
    start_epoch = end_epoch
    end_epochs.append(end_epoch)

  # create a 1-based epoch list for plotting that includes the last end_epoch
  epochs = [x for x in range(1, end_epochs[-1] + 1)]

  # Diagnostics
  # print(epochs)
  # print(loss_train)
  # print(end_epochs)

  # Make plots
  figure = plt.figure(figsize=figsize)
  plt.subplot(2, 1, 1)
  plt.plot(epochs, loss_train, label='Training Loss')
  plt.plot(epochs, loss_val, label='Validation Loss')
  # plt.legend(loc='upper right')
  plt.legend()
  plt.title('Training and Validation Loss')
  # Plot vertical lines for changes in epochs
  for epoch in end_epochs:
    plt.plot([epoch, epoch], plt.ylim())

  plt.subplot(2, 1, 2)
  plt.plot(epochs, metric_train, label='Training ' + metric)
  plt.plot(epochs, metric_val, label='Validation ' + metric)
  # plt.legend(loc='lower right')
  plt.legend()
  plt.title('Training and Validation ' + metric)
  plt.xlabel('epoch')
  for epoch in end_epochs:
    plt.plot([epoch, epoch], plt.ylim())
  plt.show()

  return figure


def scatter_plot(y_actual, y_predicted, title=None):
  """
  Plot y actual y modeled with a 1:1 reference line

  Args:
    y_actual:
      true y labels
    y_predicted:
      predicted y labels
    title (str|None):
      figure title

  """
  plt.scatter(y_actual, y_predicted, marker='x')
  y_max = tf.reduce_max(y_actual)
  plt.plot([0, y_max], [0, y_max], 'g.-')
  plt.xlabel('y True')
  plt.ylabel('y Predicted')
  if title:
    plt.title(title)


def plot_predictions(train_data,
                     train_labels,
                     test_data=None,
                     test_labels=None,
                     predictions=None,
                     title=None):
  """
  Plots training data, test data and compares predictions.

  Args:
    train_data:
      X (features) training data
    train_labels:
      y (labels) training data
    test_data:
      X (features) test data
    test_labels:
      y (labels) test data
    predictions:
      y (labels) model predictions from test data
    title (str|None):
      figure title
  """
  # plt.figure(figsize=(10, 7))
  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training")
  # Plot test data in green
  if test_data is not None:
    plt.scatter(test_data, test_labels, c="r", label="Testing")
  # Plot the predictions in red (predictions were made on the test data)
  if predictions is not None:
    plt.scatter(test_data, predictions, c="g", label="Predictions", marker='x')
  # Show the legend
  plt.legend()
  if title is not None:
    plt.title(title)


def pred_and_plot2(model,
                   file_names,
                   class_names,
                   img_shape,
                   actual_class_names=None,
                   rescale=255.,
                   ):
  """
  Predict and plot classes associated with image files based on
  a tf model.

  Args:
    model:
      tf model to make class predictions based on image files
    file_names:
      file names of images to classify with the tf model
    class_names list(str):
      Class names in alphabetical order.
    img_shape (int):
    actual_class_names list(str):
      Class names associated with each file.
    rescale (float):
    Scaling factor because some models expect tensor data between 0 and 1 and others 0 and 255
    rescale = 1. if your model expects values between 0 and 255 (i.e. you are not rescaling)
    rescale = 255. if your model expects values between 0 and 1 (you are rescaling the photos by normalization).

  Notes:
    * ensure that img_shape is selected to be consistent with expected model input.
    * file_names, classes = tfu.random_sample_all_classes(testing_directory, number_files_per_subdirectory)
    * class_names = tfu.dir_class_names(training_directory)
  """
  # If passed a single file_name, pack it in a list
  if isinstance(file_names, str):
    file_names = [file_names]

  # Import the target image and preprocess it
  images = image_files_to_tensor(file_names, img_shape=img_shape, rescale=rescale)

  # Make predictions
  predictions = model.predict(images)

  if actual_class_names is None:
    actual_class_names = [None] * len(images)
  elif isinstance(actual_class_names, str):
    actual_class_names = [actual_class_names] * len(images)

  # Visualize all predictions
  for image, prediction, actual in zip(images, predictions, actual_class_names):
    # print(type(prediction), prediction.shape, prediction[0])

    if len(class_names) == 2:
      # print(f"Binary Data")
      max_prediction = prediction[0]
      # Prediction is the probability that the item is the second in the class
      # If it is the 1st in the class, you need to change percentage meaning
      if max_prediction > 0.5:
        class_index = 1
      else:
        class_index = 0
        max_prediction = 1.0 - max_prediction
    elif len(class_names) > 2:
      # print(f'Multiclass Data')
      class_index = prediction.argmax()
      # print(f"class_index: {class_index}")
      max_prediction = prediction[class_index]
    else:
      raise ValueError(f'{class_names} must be 2 or more in length')

    pred_class_name = class_names[class_index]

    title = f"Prediction: {pred_class_name} ({max_prediction * 100:.2f} %)"
    if actual:
      title = actual + " " + title

    # Plot the image and predicted class
    plt.figure()
    # imshow expects images to be floats between 0 and 1 or ints between 0 and 255
    plt.imshow(image * rescale / 255.)
    plt.title(title)
    plt.axis(False)


def plot_scaled(X_train_standard,
                X_test_standard,
                X_train_norm,
                X_test_norm,
                y_train,
                y_test):
  """
  Plot training and test data that has been normalized (scaled between 0 and 1)
  and standardized (0 mean with unit variance)

  Args:
    X_train_standard:
      X training data standardized
    X_test_standard:
      X test data standardized
    X_train_norm:
      X training data normalized
    X_test_norm:
      X test data normalized
    y_train:
      y training data
    y_test:
      y test data
  """
  # plt.figure(figsize=(10, 7))

  plt.scatter(X_train_standard, y_train, c="b", label="train_standard")
  plt.scatter(X_test_standard, y_test, c="r", label="test_standard")
  plt.scatter(X_train_norm, y_train, c="g", label="train_norm")
  plt.scatter(X_test_norm, y_test, c="y", label="test_norm")

  plt.legend()
  plt.title('Normalized data')


def make_confusion_matrix(y_true,
                          y_pred,
                          classes=None,
                          figsize=(10, 10),
                          text_size=15,
                          normalize='true',
                          errors_only=False,
                          title=None,
                          ):
  """
  Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    normalize : {'true', 'pred'}
        Normalizes confusion matrix over the true (rows), predicted (columns).
    errors_only: True if you want to remove correct predictions (along the diagonal)
        and only focus on the errors.
    title: plot title for confusion matrix

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example:
    make_confusion_matrix(y_true=test_labels,
                          y_pred=y_preds,
                          classes=class_names,
                          figsize=(15, 15),
                          text_size=10)

  Notes:
    1) This function is a modification of sklearn's plot_confusion_matrix
    2) plot_confusion_matrix function
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
    3) ML introductory notebook
        https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
  """
  # Remove correct values on diagonal if desired
  if errors_only is True:
    sample_weight = y_true != y_pred
  else:
    sample_weight = None

  # If classes in dictionary form, cast them to an array
  if classes and isinstance(classes, dict):
    classes = [class_name for class_name in classes.values()]

  # Create the confusion matrix
  cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)

  if normalize == 'true':
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]  # normalize it col
  elif normalize == 'pred':
    cm_norm = cm.astype("float") / cm.sum(axis=0)  # normalize it by row
  else:
    raise ValueError(f"{normalize} is unknown normalization")
  n_classes = cm.shape[0]  # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues)  # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  if title is None:
    title = "Confusion Matrix"

  ax.set(title=title,
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes),  # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels,  # axes labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    text = f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)"
    plt.text(j, i, text,
             horizontalalignment="center",
             color="white" if cm[i, j] > threshold else "black",
             size=text_size)


def plot_image(file_name, title=None, include_shape=True, figsize=None):
  """
  Plot image file with title and optional image shape.

  Args:
    file_name (str | Path):
      filename to plot
    title (str|None):
      figure title
    include_shape (bool):
      True if you want the figure shape to be in the figure title
    figsize:
      figure_size for plt figure

  """
  image = mpimg.imread(file_name)
  return plot_tensor(image, title=title, include_shape=include_shape, figsize=figsize)


def plot_tensor(x, title=None, include_shape=True, figsize=None):
  """
  Plot image file with title and optional image shape.

  Args:
    x:  tensor to visualize
    title (str|None): figure title
    include_shape (bool):
      True if you want the figure shape to be in the figure title
    figsize:
      figure_size for plt figure
  """
  # Convert to numpy if needed/possible
  try:
    x = x.numpy()
  except AttributeError as e:
    pass
    # print(f"{type(x) = } does not have .numpy() method")
    # print(f"{type(e) = }, {e = }")

  # image routines expect integers
  # Use squeeze in-case the image was grayscale (1 channel)
  img = np.squeeze(x.astype('uint8'))

  if figsize is None:
    fig = plt.figure()
  else:
    fig = plt.figure(figsize=figsize)
  plt.imshow(img)
  plt.axis("off")
  if include_shape:
    if title:
      title += f' Shape: {img.shape}'
    else:
      title = f'Shape: {img.shape}'
  if title:
    plt.title(title)
  plt.show()
  return fig, img


def plot_single_class(directory, class_name, number_files=1, figsize=None):
  """
  Plot random images for a single class.

  Args:
    directory (str|Path): directory to sample files from
    class_name (str): image class of interest
    number_files (int): number of files to return
    figsize: figure_size for plt figure
  """
  image_paths = random_sample_single_class(directory, class_name, number_files)
  # print(f"image_path= {image_paths}")

  for image_path in image_paths:
    # Read in the image and plot it using matplotlib
    title = f'Class: {class_name}'
    _fig, _img = plot_image(image_path, title, figsize=figsize)
  return image_paths


def plot_all_classes(directory, number_files, figsize=None):
  """
  Plot sampled files from a directory for each image class type.

  Args:
    directory (str|Path): directory to sample files from
    number_files (int): number of files to return for each class type
    figsize: figure_size for plt figure
  """
  print(f'Plotting {number_files} file of each class from: {directory}')
  file_names, classes = random_sample_all_classes(directory, number_files)

  for file_name, class_name in zip(file_names, classes):
    title = f'Class: {class_name}'
    _fig, _img = plot_image(file_name, title, figsize=figsize)
  return file_names, classes


def plot_augmented_image(file_name, title, **kwargs):
  """
  Plot 9 augmented images of file_name given augmentations in kwargs

  Args:
    file_name: file to be augmented/visualized
    title: figure title
    **kwargs: passed to ImageDataGenerator
  """
  # create image data augmentation generator
  datagen = ImageDataGenerator(**kwargs)

  # load the image
  img = load_img(file_name)
  # convert to numpy array
  data = img_to_array(img)
  # expand dimension to one sample
  samples = np.expand_dims(data, 0)

  # prepare iterator
  it = datagen.flow(samples, batch_size=1)

  plt.figure()
  plt.suptitle(title)

  # generate samples and plot
  for i in range(9):
    # define subplot
    plt.subplot(330 + 1 + i)
    # generate batch of images
    batch = it.next()
    # print(f'Batch {i} is type: {type(batch)}')
    # print(f'Batch {i} shape: {batch.shape}')
    # convert to unsigned integers for viewing
    image = batch[0].astype('uint8')
    # plot raw pixel data
    plt.imshow(image)
  # show the figure
  plt.show()


###############################################################################
# Fitting and callbacks
###############################################################################
def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instance to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


###############################################################################
# Model Evaluation & Performance
###############################################################################

def random_guess_accuracy(test_labels):
  """
  Calculate the accuracy of a random reshuffling of label categories.

  Useful in setting a baseline to compare model performance against.

  Args:
    test_labels: categorical labels associated with test data

  Returns:
    random_accuracy (float) : random reshuffle accuracy

  Notes:
    *  If you have very skewed data (99% negative, 1% positive),
       random reshuffle accuracy may be high (e.g. 99%).

  """
  test_labels_copy = copy.copy(test_labels)
  np.random.shuffle(test_labels_copy)
  is_same_label = np.array(test_labels) == np.array(test_labels_copy)
  random_accuracy = is_same_label.mean()
  print(f'The accuracy of randomly shuffle guessing labels is {random_accuracy * 100:.2f} %')
  return random_accuracy


def baseline_directory_accuracy(test_directory, class_mode='binary'):
  """
  Calculate the accuracy of a random reshuffling of label categories of a test directory.

  Args:
    test_directory: directory with test data
    class_mode (str):
      Can be 'binary' or 'categorical' depending on type of classification problem

  Returns:
    random_accuracy (float) : random reshuffle accuracy

  Notes:
    *  If you have very skewed data (99% negative, 1% positive),
       baseline_directory_accuracy accuracy may be high (e.g. 99%).
  """
  y_labels, counts = generator_labels(test_directory, class_mode='binary')
  return random_guess_accuracy(y_labels)


def model_evaluate(model, X_test, y_test):
  """
  Evaluate a trained model on a test dataset
  using the metrics specified via model.compile metrics specification.

  Args:
    model: tf model for evaluation
    X_test: X feature data
    y_test: y label data

  Returns:
    results of model evaluation on test data
  """
  evaluation = model.evaluate(X_test, y_test)
  print(f'Model evaluation: {evaluation}')
  return evaluation


def mae_mse_metrics(y_test, y_preds):
  """
  Calculate mean_absolute_error (mae) and mean_squared_error (mse) metrics of model performance

  Args:
    y_test: y label test values
    y_preds: y label predicted values

  Returns:
    (mae, mse)
  """

  # Note, you likely have to squeeze the tests and predictions
  # for the metric function to work as expected.
  mae = tf.metrics.mean_absolute_error(y_true=tf.squeeze(y_test),
                                       y_pred=tf.squeeze(y_preds))
  mse = tf.metrics.mean_squared_error(y_true=tf.squeeze(y_test),
                                      y_pred=tf.squeeze(y_preds))
  print(f'MAE: {mae}, MSE: {mse}')
  return mae, mse


def predictions_to_pandas(X_train,
                          y_train,
                          X_test,
                          y_test,
                          tf_models,
                          model_names,
                          ):
  """
  Display multiple model predictions in a pandas dataframe for easy comparison

  Args:
    X_train:
      X (feature) training data
    y_train:
      y training data
    X_test:
      X (test) training data
    y_test:
      y test data
    tf_models:
      list of models to evaluate
    model_names:
      list of model names associated with 'models'

  Returns:
    DataFrame of model results
  """
  model_results = []

  for model, model_name in zip(tf_models, model_names):
    y_train_preds = model.predict(X_train, verbose=0)
    y_test_preds = model.predict(X_test, verbose=0)
    mae_train = tf.metrics.mean_absolute_error(y_true=y_train.squeeze(),
                                               y_pred=y_train_preds.squeeze())
    mae_test = tf.metrics.mean_absolute_error(y_true=y_test.squeeze(),
                                              y_pred=y_test_preds.squeeze())
    mse_train = tf.metrics.mean_squared_error(y_true=y_train.squeeze(),
                                              y_pred=y_train_preds.squeeze())
    mse_test = tf.metrics.mean_squared_error(y_true=y_test.squeeze(),
                                             y_pred=y_test_preds.squeeze())

    # print(mae_train, mae_test, mse_train, mse_test)
    # print(mae_train.numpy(), mae_test.numpy(), mse_train.numpy(), mse_test.numpy())

    row = [model_name,
           int(mae_train.numpy()),
           int(mae_test.numpy()),
           int(mse_train.numpy()),
           int(mse_test.numpy())]

    model_results.append(row)

  all_results = pd.DataFrame(model_results, columns=["model", "mae_train", "mae_test", "mse_train", "mse_test"])
  return all_results


###############################################################################
# Timeseries related classes and functions
###############################################################################
class TimeSeries:
  """
  Class to facilitate creating training, validation, and testing datasets
  from feature and target arrays.
  """

  def __init__(self,
               features,
               target_column=0,
               fraction_training=0.6,
               fraction_validation=0.2,
               fraction_testing=0.2,
               sampling_unit="second",
               sampling_rate=1,
               window=10,
               horizon=1,
               shuffle=True,
               batch_size=32,
               standardize=False,
               ):
    """
    Args:
        features:
          Time series of features (X) used to make predictions
          features should be 2-d
        target_column:
          Specify which column of the features is also considered to be the target
          This is useful for naive forecasting error calculation
        fraction_training:
          (0 to 1) fraction of data to be used for training
        fraction_validation:
          (0 to 1) fraction of data to be used for validation
        fraction_testing:
          (0 to 1) fraction of data to be used for testing
        sampling_unit:
          Time unit associated with measurements
        sampling_rate:
          Number of samples taken in each time unit.
          e.g. if the sampling_unit was 'hour' and the sampling_rate=6,
          this would mean that your data has 6 measurements per hour (every 10 minutes)
        window:
            Number of feature datapoints used to make a prediction (AKA sequence_length).
            The number of datapoints will only equal the indices if the sampling_rate = 1.
            e.g. if sampling_unit='hours', sampling_rate=6, and window=2*24, this
            corresponds to 48 sampling points over 48 hours.
            e.g. if sampling_unit='hours', sampling_rate=1, and window=3*24, this
            corresponds to 72 sampling points over 72 hours.
        horizon:
          the number of sampling units to predict into future (from the last sampling in the window)
          e.g. if the sampling_unit='hour' and horizon=24, then the prediction
          will be 24 hours after the last sample in the window.
        shuffle:
          passed to timeseries_dataset_from_array for the training dataset
          shuffle is False for validation and testing
        batch_size:
          passed to timeseries_dataset_from_array
        standardize:
          True to change feature mean's to zero with std of 1
          Note this will change the underlying array, so you may want to
          pass a copy of the array rather than the original.

    Notes
    1) MAKE SURE YOUR FIRST TWO LAYERS SPECIFY THE INPUT AND FLATTEN
       IF YOU ARE GOING TO PASS DIRECTLY TO A DENSE LAYER
       For example
        inputs = keras.Input(shape=<TimeSeries object>.input_shape)
        x = layers.Flatten()(inputs)

    """
    # Class level variable initialization
    self.delay = None
    self.features_mean = None
    self.features_std = None
    self.input_shape = None
    self.last_in_sequence = None
    self.num_test_samples = None
    self.num_test_samples_usable = None
    self.num_train_samples = None
    self.num_train_samples_usable = None
    self.num_val_samples = None
    self.num_val_samples_usable = None
    self.samples_forward = None
    self.standardized_features_mean = None
    self.standardized_features_std = None
    self.targets = None
    self.test_dataset = None
    self.test_end_index = None
    self.test_start_index = None
    self.train_dataset = None
    self.train_end_index = None
    self.train_start_index = None
    self.usable = None
    self.val_dataset = None
    self.val_end_index = None
    self.val_start_index = None
    self.y_naive = None
    self.y_naive_test = None
    self.y_naive_train = None
    self.y_naive_val = None
    self.y_true = None
    self.y_true_test = None
    self.y_true_train = None
    self.y_true_val = None

    if features.ndim != 2:
      raise ValueError(f'features must be a 2-d matrix. {features.shape=}')

    # Save arguments
    self.features = features
    self.target_column = target_column
    self.fraction_training = fraction_training
    self.fraction_validation = fraction_validation
    self.fraction_testing = fraction_testing
    self.sampling_unit = sampling_unit
    self.sampling_rate = sampling_rate
    self.window = window
    self.horizon = horizon
    self.shuffle = shuffle
    self.batch_size = batch_size
    self.standardize = standardize

    # Initialize class variables
    self.train_dataset_samples = None
    self.val_dataset_samples = None
    self.test_dataset_samples = None
    # naive abs calculated using Chollet's dataset algorithm (slower)
    self.train_naive_abs = None
    self.val_naive_abs = None
    self.test_naive_abs = None
    # naive abs calculated with custom algorithm (fast)
    self.all_naive_abs_fast = None
    self.train_naive_abs_fast = None
    self.val_naive_abs_fast = None
    self.test_naive_abs_fast = None

    # Routines to prepare dataset for modeling
    self.num_samples = len(features)
    self.find_shape_and_delay()
    self.find_splits()
    self.find_targets()
    self.find_mean_std()
    self.create_datasets()
    self.calc_all_naive_abs_errors_2()

  def find_shape_and_delay(self):
    """
    Find the appropriate shape for use the datasets in model specification and
    determine delay input indices and target values (which will determine
    the max number of usable samples to make a forcast).

    Delay is the number to add to the first feature index to get the target index.
    Delay is used in the indexing of feature and targets as follows:
        data    = features[:-delay],
        targets = features[delay:]

        shape=(sequence_length, features.shape[-1])
    """
    # shape is (sequence length, # of features per sample)
    self.input_shape = (self.window, self.features.shape[-1])

    self.last_in_sequence = self.sampling_rate * (self.window - 1)  # Last sampling point in input sequence
    self.samples_forward = self.sampling_rate * self.horizon  # Number of samples forward from last sample
    self.delay = self.last_in_sequence + self.samples_forward
    # Alternative delay formulation
    # delay = sampling_rate * (window - 1 + horizon)

  def find_splits(self):
    """Determine training, validation, testing splits"""
    tolerance = 1e-6  # rounding tolerance
    if abs(1.0
           - self.fraction_training
           - self.fraction_validation
           - self.fraction_testing) > tolerance:
      print(f"{self.fraction_training=}, "
            f"{self.fraction_validation=}, "
            f"{self.fraction_testing=}")
      raise ValueError(f"Fractions must sum to 1.0")

    # Find maximum number of samples in each splitting group
    # Actual number of samples may be smaller due to
    # delay and windowing effectively reducing the dataset size
    self.num_train_samples = int(self.fraction_training * self.num_samples)
    self.num_val_samples = int(self.fraction_validation * self.num_samples)
    # Have test sample be the residual in case of rounding issues
    self.num_test_samples = self.num_samples - self.num_train_samples - self.num_val_samples

    # Find cut points for the datasets accounting for some of them being unusable.
    # Usable is the number of possible forecasts that can be made from the dataset.
    # Usable takes into account that you need a full window to make a projection
    # which is not possible for the data at the end of the dataset
    # The feature and target arrays will be offset by self.delay
    # The cut points below ensure that the array bounds will not be exceeded on the offset arrays
    self.usable = self.num_samples - self.delay

    # Training datasets
    self.train_start_index = 0
    self.train_end_index = min(self.usable, self.num_train_samples)
    self.num_train_samples_usable = self.train_end_index - self.train_start_index

    # Validation datasets
    self.val_start_index = min(self.usable, self.num_train_samples)
    self.val_end_index = min(self.usable, self.num_train_samples + self.num_val_samples)
    self.num_val_samples_usable = self.val_end_index - self.val_start_index

    # Testing datasets
    self.test_start_index = min(self.usable, self.num_train_samples + self.num_val_samples)
    self.test_end_index = min(self.usable, self.num_train_samples + self.num_val_samples + self.num_test_samples)
    self.num_test_samples_usable = self.test_end_index - self.test_start_index

  def find_mean_std(self):
    """
    find data mean and std.  Normally you will want to standardize based on training data
    (which is the default here), but you have the option to standardize with other datasets.
    """
    # print(f"{self.features=}")
    # print(f"{self.num_train_samples=}")
    #
    if self.standardize == "all data":
      print(f"Calculating feature means and std using all data")
      self.features_mean = self.features.mean(axis=0)
      self.features_std = self.features.std(axis=0)
    elif self.standardize == "testing":
      print(f"Calculating feature means and std using testing data")
      self.features_mean = self.features[self.test_start_index:].mean(axis=0)
      self.features_std = self.features[self.test_start_index:].std(axis=0)
    else:  # default is to use training data
      print(f"Calculating feature means and std using training data")
      self.features_mean = self.features[:self.num_train_samples].mean(axis=0)
      self.features_std = self.features[:self.num_train_samples].std(axis=0)

    if self.standardize is not False:
      self.features -= self.features_mean
      self.features /= self.features_std

      self.standardized_features_mean = self.features[:self.num_train_samples].mean(axis=0)
      self.standardized_features_std = self.features[:self.num_train_samples].std(axis=0)

  def create_dataset(self, start_index, end_index, shuffle=False):
    """Create a single dataset (if possible) given a start and end index"""

    if start_index < end_index:
      if end_index >= self.usable:
        end_index = None
      return keras.utils.timeseries_dataset_from_array(
        data=self.features[:-self.delay],
        targets=self.targets[self.delay:],
        sampling_rate=self.sampling_rate,
        sequence_length=self.window,
        shuffle=shuffle,
        batch_size=self.batch_size,
        start_index=start_index,
        end_index=end_index,
      )
    else:
      return None

  def create_datasets(self):
    """
    Create datasets from timeseries and metadata.

    Notes:
      1) The timeseries_dataset_from_array api formulation appears to use None
      if the end index is >= the length of the data
    """
    # Only shuffle the training_dataset
    self.train_dataset = self.create_dataset(self.train_start_index,
                                             self.train_end_index,
                                             self.shuffle)

    self.val_dataset = self.create_dataset(self.val_start_index,
                                           self.val_end_index)

    self.test_dataset = self.create_dataset(self.test_start_index,
                                            self.test_end_index)

  def diagnostics(self, verbose=0):
    """
    Print diagnostics for data inspection
    Args:
      verbose:
      0: min output
      1: average output
      2: max output
    """
    print(f"Sample Splits\n{'-' * 80}")
    print(f"Total Samples     (100.0%) {self.num_samples}")
    print(f"Total Usable      ({100 * self.usable / self.num_samples:5.1f}%) {self.usable}")
    print(f"Training Total    ({100 * self.fraction_training:5.1f}%) {self.num_train_samples}")
    print(
      f"Training Usable   ({100 * self.num_train_samples_usable / self.num_samples:5.1f}%) {self.num_train_samples_usable}")
    if self.train_dataset_samples:
      print(
        f"Training Used     ({100 * self.train_dataset_samples / self.num_samples:5.1f}%) {self.train_dataset_samples}")
    print(f"Validation Total  ({100 * self.fraction_validation:5.1f}%) {self.num_val_samples}")
    print(
      f"Validation Usable ({100 * self.num_val_samples_usable / self.num_samples:5.1f}%) {self.num_val_samples_usable}")
    if self.val_dataset_samples:
      print(f"Validation Used   ({100 * self.val_dataset_samples / self.num_samples:5.1f}%) {self.val_dataset_samples}")
    print(f"Testing Total     ({100 * self.fraction_testing:5.1f}%) {self.num_test_samples}")
    print(
      f"Testing Usable    ({100 * self.num_test_samples_usable / self.num_samples:5.1f}%) {self.num_test_samples_usable}")
    if self.test_dataset_samples:
      print(
        f"Testing Used      ({100 * self.test_dataset_samples / self.num_samples:5.1f}%) {self.test_dataset_samples}")

    print(f"{self.sampling_unit = }")
    print(f"{self.sampling_rate = }")
    print(f"{self.window = }")
    print(f"{self.horizon = }")
    print(f"{self.shuffle = }")
    print(f"{self.batch_size = }")
    print(f"Delay is the offset between feature index and target index.")
    print(f"{self.delay = }")
    self.dataset_summary(self.train_dataset, "Training Dataset", verbose)
    self.dataset_summary(self.val_dataset, "Validation Dataset", verbose)
    self.dataset_summary(self.test_dataset, "Test Dataset", verbose)

    print(f"Features mean and std before potential normalization")
    rounded_mean = self.features_mean.copy()
    rounded_mean[rounded_mean < 1e-6] = 0
    print(f"Feature Means\n{rounded_mean}")
    print(f"Feature STDs\n{self.features_std}")

    if self.standardize is True:
      print(f"Features mean and std after normalization")
      rounded_mean = self.standardized_features_mean.copy()
      rounded_mean[rounded_mean < 1e-6] = 0
      print(f"Feature Means\n{rounded_mean}")
      print(f"Feature STDs\n{self.standardized_features_std}")

    print(f"\nNaive ABS Error for datasets using Chollet's algorithm")
    if self.train_naive_abs:
      print(f"Training   Naive ABS: {self.train_naive_abs:.3f}")
    else:
      print(f"Training   Naive ABS: (Not Calculated Yet)")
    if self.val_naive_abs:
      print(f"Validation Naive ABS: {self.val_naive_abs:.3f}")
    else:
      print(f"Validation Naive ABS: (Not Calculated Yet)")
    if self.test_naive_abs:
      print(f"Testing    Naive ABS: {self.test_naive_abs:.3f}")
    else:
      print(f"Testing    Naive ABS: (Not Calculated Yet)")

    print(f"\nNaive ABS Error for datasets using fast calculation algorithm")
    if self.all_naive_abs_fast:
      print(f"All Data   Naive ABS: {self.all_naive_abs_fast:.3f}")
      if self.train_naive_abs_fast:
        print(f"Training   Naive ABS: {self.train_naive_abs_fast:.3f}")
      if self.val_naive_abs_fast:
        print(f"Validation Naive ABS: {self.val_naive_abs_fast:.3f}")
      if self.test_naive_abs_fast:
        print(f"Testing    Naive ABS: {self.test_naive_abs_fast:.3f}")

  def dataset_summary(self, dataset, name, verbose=0):
    """
    Display summary of dataset
    Args:
      dataset:
      name:
      verbose:
    """
    if dataset is None:
      print(f"\n{name}: dataset is empty")
      return
    else:
      print(f"\n{name}: # of batches = {len(dataset)}")

    if verbose >= 1:
      count = 0
      for i, (x_inputs, y_targets) in enumerate(dataset):
        if verbose >= 2:
          print(f'Batch #{i}.  X.shape: {x_inputs.shape}, y.shape: {y_targets.shape}')
        for j, (x, y) in enumerate(zip(x_inputs, y_targets)):
          count += 1
          if verbose >= 2:
            print(f"\t Sub-batch {j}:\t{np.squeeze(x)} --> {y}")
      print(f"Number of records: {count}")

  def calc_all_naive_abs_errors_1(self):
    """Find all naive abs errors using datasets"""
    self.train_dataset_samples, self.train_naive_abs = self.naive_abs_error_1('train',
                                                                              self.train_dataset)
    self.val_dataset_samples, self.val_naive_abs = self.naive_abs_error_1('validation',
                                                                          self.val_dataset)
    self.test_dataset_samples, self.test_naive_abs = self.naive_abs_error_1('test',
                                                                            self.test_dataset)

  def calc_all_naive_abs_errors_2(self):
    """
    Calculate the mean_abs_err for train, val, and test data without using datasets.

    MAE error using offset method rather than deep learning approach (using datasets)
    Is faster than deep learning book algorithm, but with slightly diff results
    """

    self.all_naive_abs_fast = self.naive_abs_error_2("All data   ", 0, self.num_samples)
    self.train_naive_abs_fast = self.naive_abs_error_2("Training   ", self.train_start_index, self.train_end_index)
    self.val_naive_abs_fast = self.naive_abs_error_2("Validation ", self.val_start_index, self.val_end_index)
    self.test_naive_abs_fast = self.naive_abs_error_2("Testing    ", self.test_start_index, self.test_end_index)

  def naive_abs_error_1(self,
                        name,
                        dataset,
                        verbose=0):
    """
    Calculate the mean absolute error using a naive approach
    where the forecasted target is simply equal to the last feature value in the window.

    This method was in Chalets Deep Learning book.  It is slow, but easy to understand.

    Args:
      name:
      dataset:
      verbose:

    Notes:
      1) features[:, -1, i]   indexing explained
                  :             -> take all rows in batch
                     -1         -> take the last sample in the sequence
                                   as the prediction for the target
                                   (the next guess is the current value)
                         i      -> make prediction using only the feature column
                                   that is associated with the target
      2) input may be standardized, but the targets are never standardized
         if the input was standardized, to compare input to targets,
         the feature data must be un-standardize to compare to the targets
         using the stdev and mean
    """
    if dataset is None:
      return None, None

    i = self.target_column  # the index of target value in the feature matrix
    feature_mean = self.features_mean[i]
    feature_std = self.features_std[i]

    total_abs_err = 0.
    num_samples = 0

    for features, targets in dataset:
      predictions = features[:, -1, i]
      if predictions.shape != targets.shape:
        print(f"{predictions.shape = }, {targets.shape = }")
        raise ValueError(f"Predictions and targets are not the same shape,"
                         f" perhaps one needs to be squeezed?")
      if self.standardize is True:
        predictions = predictions * feature_std + feature_mean
      total_abs_err += np.sum(np.abs(predictions - targets))
      num_samples += len(features)

    mean_abs_err = total_abs_err / num_samples
    print(f"{name}: Number of samples processed = {num_samples}")

    return num_samples, mean_abs_err

  def naive_abs_error_2(self,
                        name,
                        start_index,
                        stop_index,
                        verbose=0
                        ):
    """
    Calc single mean_abs_err given a start and stop index (without using datasets).
    Note that datasets may not use the maximum usable data points, so results
    may be slightly different from naive_abs_error_1, but should be quicker.

    """
    if start_index < stop_index:
      mae = tf.keras.metrics.mean_absolute_error(self.y_true[start_index:stop_index],
                                                 self.y_naive[start_index:stop_index])
      if verbose > 0:
        print(f"{name} mae calculated to be: {mae:.3f}")
      return mae

  def find_targets(self):
    """
    Determine targets, y_true, and y_naive from feature array.

    Note that y_true may be longer than the dataset created from timeseries_dataset_from_array
    because of rounding/windowing issues.

    Formulation Approach:
    The naive approach uses the last value in the window to predict the target.
    y_true = targets[delay:]
    y_pred = targets[last_in_sequence: - (delay - last_in_sequence) ]

    if sample_rate is 1, the window is 1, and the horizon is 1,
    the last_in_sequence = 0, the delay = 1, and the indexing reduces to:
    targets = features[1:]
    naive = features[:-1]
    """
    # Determine the target data from features
    # The targets are a copy of features, so they don't accidentally
    # get scaled during standardization
    self.targets = self.features[:, self.target_column].copy()
    self.y_true = self.targets[self.delay:]
    self.y_naive = self.targets[self.last_in_sequence:-(self.delay - self.last_in_sequence)]

    # find the true and naive associated with the splits
    self.y_true_train = self.y_true[self.train_start_index: self.train_end_index]
    self.y_naive_train = self.y_naive[self.train_start_index: self.train_end_index]
    self.y_true_val = self.y_true[self.val_start_index: self.val_end_index]
    self.y_naive_val = self.y_naive[self.val_start_index: self.val_end_index]
    self.y_true_test = self.y_true[self.test_start_index: self.test_end_index]
    self.y_naive_test = self.y_naive[self.test_start_index: self.test_end_index]

  def evaluate_metrics(self,
                       y_true,
                       y_pred,
                       scaling='all data'):
    """
    Evaluate predictions using a variety of metrics.

    Notes
     1) datasets typically have less datapoints than the total possible
     so the length of predictions may be less than the length of true.
     2) mean absolute scaled error is scaled by 'all data' by default and can also be
     scaled by 'testing'.
     3) mean_absolute_percentage_error may blow up if targets are ever zero.
    """
    y_true = tf.squeeze(y_true)
    y_pred = tf.squeeze(y_pred)

    num_true = len(y_true)
    num_pred = len(y_pred)
    num_min = min(num_true, num_pred)  # use the minimum datapoints in common

    if num_true != num_pred:
      print(f"Warning, true and prediction lengths are not the same, truncating to predictions")
      print(f"{y_true.shape = }, {y_pred.shape = }")

    # Make sure float64 (for metric calculations)
    y_true = tf.cast(y_true[:num_min], dtype=tf.float64)
    y_pred = tf.cast(y_pred[:num_min], dtype=tf.float64)

    # Calculate various metrics
    mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
    mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)  # puts and emphasis on outliers (all errors get squared)
    rmse = tf.sqrt(mse)
    mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
    # print(mae, self.test_naive_abs_fast, self.all_naive_abs_fast)
    if scaling == 'testing':
      mase = mae / tf.cast(self.test_naive_abs_fast, dtype=tf.float64)
    else:  # default is to use all data for scaling
      mase = mae / tf.cast(self.all_naive_abs_fast, dtype=tf.float64)

    return {"mae": mae.numpy(),
            "mse": mse.numpy(),
            "rmse": rmse.numpy(),
            "mape": mape.numpy(),
            "mase": mase.numpy()}

  def naive_metrics(self, dataset='all data'):
    """
    Calculate various metrics using naive predictions.
    """
    print(f"Calculating metrics using dataset: {dataset}")
    if dataset == "all data":
      y_true = self.y_true
      y_naive = self.y_naive
    elif dataset == "training":
      y_true = self.y_true_train
      y_naive = self.y_naive_train
    elif dataset == "validation":
      y_true = self.y_true_val
      y_naive = self.y_naive_val
    elif dataset == "testing":
      y_true = self.y_true_test
      y_naive = self.y_naive_test
    else:
      raise ValueError(f"{dataset} is an unknown dataset type")

    return self.evaluate_metrics(y_true, y_naive, scaling=dataset)

  def make_predictions(self, model):
    results = model.evaluate(self.test_dataset)
    print(f"model.evaluate results: {results}")
    preds = model.predict(self.test_dataset)
    print(f"model.predict preds.shape: {preds.shape}")
    return results, preds

  def plot_test_vs_real(self, model):
    """
    Plot actual test data versus predicted test data.
    Note: todo - figure out why plotting data needs an offset to work!
    this is a bit of a hack that needs to be made explicit

    Args:
      model:

    Returns:

    """
    results = model.evaluate(self.test_dataset)
    print(f"model Results: {results}")
    print(f"model Val MAE: {results[1]:.3f}")

    preds = model.predict(self.test_dataset)
    metrics = self.evaluate_metrics(self.y_true_test, preds)
    print(metrics)

    plot_offset = len(self.y_true_test) - len(preds)
    i_start = self.test_start_index + plot_offset
    i_end = i_start + len(preds)
    x = list(range(i_start, i_end))
    y = self.features[i_start:i_end, self.target_column]
    if self.standardize is True:
      y *= self.features_std[self.target_column]
      y += self.features_mean[self.target_column]

    plt.plot(x, y, '-', label='true')
    plt.plot(x, preds, '-', label='pred')
    plt.legend()


###############################################################################
# Time Series Windowing using googles analysis technique
# https://www.tensorflow.org/tutorials/structured_data/time_series
###############################################################################


###############################################################################
# Ensemble modeling
###############################################################################
def create_preds_dict(model_names, models_dict, test_dataset):
  """
  Create a predictions dictionary.
  next use create_preds_array then ensemble_preds
  Args:
    model_names:
    models_dict:
    test_dataset:

  Returns:

  """
  preds_dict = {}
  for model_name in model_names:
    print(f"{model_name=}")
    model = models_dict[model_name]
    preds = model.predict(test_dataset)
    preds_dict[model_name] = preds
  return preds_dict


def create_preds_array(model_names, preds_dict):
  """
  create a 2-d array of predictions from predictions dictionary
  """
  all_preds = []
  for model in model_names:
    preds = preds_dict[model]
    # print(f"{len(preds)=}")
    all_preds.append(preds)

  # Convert a list of lists to a 2-d numpy array with
  preds_array = np.hstack(all_preds)
  print(f"{preds_array.shape=}")

  return preds_array


def ensemble_preds(preds, method='median'):
  """
  preds is 2-d numpy array with shape = (number of predictions per model, number of models)
  preds was likely created using np.hstack
  """
  if method == 'median':
    ensemble = np.median(preds, axis=1)
  elif method == 'mean':
    ensemble = np.mean(preds, axis=1)
  else:
    raise ValueError(f"Unknown ensemble method: {method}")

  print(f"Using {method} to create an ensemble average of data with shape: {ensemble.shape=}")
  return ensemble


###############################################################################
# Misc Utility functions and classes
###############################################################################
def min_max_mean(x, caption=""):
  """
  Output the min, max, mean of an array
  Args:
    x (ArrayLike):
    caption (str): Display caption for output
  """
  print(f'\n{caption}\n{"-" * 20}')
  print(f'min  ', np.min(x, 0))
  print(f'max  ', np.max(x, 0))
  print(f'mean ', np.mean(x, 0))


def ensure_flag(flags, required, conditional):
  """
  Ensure that a conditional flag can't be true if a flag it requires is False.
  Args:
    flags (dict): dictionary of flags.
    required (str): flag required to be true by conditional.
    conditional (str): flag that can only be True if requires is True.
  """
  if flags[conditional] is True:
    if flags[required] is False:
      print(f"{conditional} is reset to False because required flag {required} is not True")
      flags[conditional] = False


class Timer:
  """
  Simple timer class to evaluate Jupyter Notebook run times.
  """
  _start_time = datetime.now()

  @staticmethod
  def start(memo="Starting Notebook at time:"):
    current_time = datetime.now()
    Timer._start_time = current_time
    time_str = current_time.strftime("%H:%M:%S")
    print(f"{memo} {time_str}")

  @staticmethod
  def lap(memo=""):
    current_time = datetime.now()
    time_str = current_time.strftime("%H:%M:%S")
    print(f"Current time: {time_str}")
    print(f"Elapsed time ({memo}): {current_time - Timer._start_time}")


def mount_google_drive():
  """
  Determine if you are running on a Google colab environment or a local machine using an
  IDE such as pycharm.

  Returns:
    run_mode (string): 'colab' | 'pycharm'

  Notes:
  1) If you can mount a gdrive, you are in colab, otherwise you are in a local environment
  such as pycharm.
  2) A mounted drive may be in a different data center than the colab VM,
  so data access may be considerably slower than direct access of the VM drive,
  however files saved on a mounted drive will persist after the end of the colab session.

  Tutorials:

  * Tutorial notebook that provides recipes for loading and saving data from external sources:
    - https://colab.research.google.com/notebooks/io.ipynb
  * Tutorial on large external data access speeds from Google Colab notebooks:
    - https://ostrokach.gitlab.io/post/google-colab-storage/
  """

  try:
    # noinspection PyUnresolvedReferences
    from google.colab import drive

    run_mode = 'colab'
    content_dir = "/content/drive/"
    drive.mount(content_dir)

    # Google Drive will mount at: /content/drive/MyDrive
    print(f'\nGoogle Drive Contents:')
    os.system(f'ls {content_dir}/MyDrive')

  except ModuleNotFoundError:
    run_mode = 'pycharm'

  print(f'Running notebook on platform: {run_mode}')
  return run_mode


def find_file_structure(run_mode):
  """
  Determine the file structure for ML input/output based on run_mode.

  Args:
    run_mode (string): 'colab' | 'pycharm'
       determined from find_run_mode_and_mount_google_drive()

  Returns:
    paths (dict):
      dictionary of file paths

  Notes:
  *  Most important keys of paths:
       'dir_project': location of python project files (e.g. *.py, *.ipynb, etc)
                      On windows this will be on C:
                      On colab it will be the default home directory location
       'dir_util': ml_util package with utility files such as tf_util.py
       'dir_local': large files that should not be backed up (images, etc)
       'dir_model_runs': model run checkpoints and saves
       'dir_tensor_board': tensor board related files
       'dir_images_local': images that will be on a local drive and not backed up on g_drive
                           note, using local drive for large image collections is much faster than using remote images
    *  If you are running on Colab, the mounted Gdrive may be very slow
       compared to the local/VM drive.
    *  If you are running on a pc or mac that backs up the hard drive with Gdrive
       images should be located on a drive that does not sync with Gdrive
       to avoid wasting bandwidth/space on image files
    *  paths['dir_project'] has some subtle difference depending on platform
       e.g., colab path uses 'MyDrive' (one-word), mac uses 'My Drive' (two-word)
    *  paths['dir_images_local'] should be used for image files to address
       the two points above
  """
  paths = {}

  if run_mode == 'colab':
    paths['dir_project'] = "/content/ml_rsdas/"
    paths['dir_local'] = "/content/local/"
  elif run_mode == 'pycharm':
    paths['dir_project'] = "C:/one_drive/code/pycharm/ml_rsdas/"
    paths['dir_local'] = "C:/local/ml_local/"
  elif run_mode == 'carb_hpc':
    paths['dir_project'] = "/data/arb/tmpRuntime/projects/rsdas_ml/ml_rsdas/"
    paths['dir_local'] = "/data/arb/tmpRuntime/projects/rsdas_ml/ml_local/"
  else:
    print(f"Unknown run mode: {run_mode}")

  ######################################################################
  # Project Directory Files
  # windows - the local file system will be used and will be fast
  # colab - g_drive will be mounted and slower than local directories
  ######################################################################
  # TF utility package (where tf_util is located)
  paths['dir_util'] = paths['dir_project'] + "ml_util/"

  ######################################################################
  # Local Files (not backed up)
  ######################################################################
  # directory for storing model run input/output
  paths['dir_model_runs'] = paths['dir_local'] + "model_runs/"
  # Storing of tensorboard model statistics
  paths['dir_tensor_board'] = paths['dir_local'] + "tensor_board/"
  # local image dir for fast file access
  paths['dir_images_local'] = paths['dir_local'] + "images/"

  # Base directory of images on the gdrive has a single subdirectory of 26 letters
  # Subdirectory with 26 letters in a single directory
  paths['dir_project_letters'] = paths['dir_images_local'] + "letters/"

  # Create the directory structure if needed
  for key, path in paths.items():
    # print(f"{key=}, {path=}")
    # print(f"\n{directory}\n{'-' * 60}")
    mk_dir(path)

  return paths


def diagnostics(paths):
  """
  Diagnostics on library versions and directory structure
  Args:
    paths (dict): path dictionary created with find_file_structure
  """
  Timer.start('TensorFlow Utilities (tfu) start time:')
  print("Python Version:", sys.version)
  print("TensorFlow Version:", tf.__version__)  # (should be 2.x+)

  print(f"\nDirectory Structure\n{'-' * 60}")
  for key, path in paths.items():
    print(f"{key} \t==>\t{path}")

######################################################################
# Module Level Variables
######################################################################
# If you want to mount your google drive, use mount_google_drive
# hard-coding run_mode here for expediency
# run_mode = mount_google_drive()
run_mode = 'carb_hpc'
paths = find_file_structure(run_mode)
diagnostics(paths)

Timer.lap(f"tf_util file loading complete at:")
