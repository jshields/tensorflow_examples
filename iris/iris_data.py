"""
Based on:
https://www.tensorflow.org/get_started/get_started_for_beginners#the_iris_classification_problem
"""
import pandas as pd
import tensorflow as tf

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
SPECIES = ['Setosa', 'Versicolor', 'Virginica']


def maybe_download():
    """
    Download dataset to local cache if not there already ( ~/.keras/datasets/)
    """
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path


def load_data(label_name='Species'):
    """
    Parses the csv file in TRAIN_URL and TEST_URL.

    With feature x, label y
    returns the iris dataset as (train_x, train_y), (test_x, test_y).
    """
    # Create a local copy of the training set.
    train_path, test_path = maybe_download()

    # train_path now holds the pathname: ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.
    train = pd.read_csv(
        train_path,
        names=CSV_COLUMN_NAMES,  # list of column names
        header=0  # ignore the first row of the CSV file.
    )
    # CSV header irrelevant to our program. Appears to contain:
    # number of examples (rows of data), number of features, the 3 species names

    # `train` now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)


def train_input_fn(features, labels, batch_size):
    """
    An input function for training

    Arguments expecting an "array" can accept nearly anything that can be converted to an array with numpy.array.
    One exception is tuple which has special meaning for Datasets:
    Datasets can represent a collection of simple arrays,
    but datasets are much more powerful than this.
    Datasets transparently handle any nested combination of dictionaries or tuples.
    https://www.tensorflow.org/get_started/datasets_quickstart#slices

    Args:
        features: A {'feature_name':array} dictionary (or DataFrame) containing the raw input features.
        labels : An array (in this case pandas Series) containing the label for each example.
        batch_size : An integer indicating the desired batch size.
    """

    """
    NOTE:
    Breakdown of `(dict(features), labels)` described here:
    https://www.tensorflow.org/get_started/datasets_quickstart

    tuple commonly represents pairs of (features, labels), e.g.
    https://www.tensorflow.org/programmers_guide/datasets#reading_input_data

    tuple is one way to achieve the form that the train method requires:
    https://www.tensorflow.org/get_started/get_started_for_beginners#train_the_model

    This appears to be common convention, and/or the form that `classifier.train` requires?

    Create a dataset containing (features, labels) pairs:
    """
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle (randomize), repeat, and batch the examples.
    """
    Training works best if the training examples are in random order.
    Setting the buffer_size to a value larger than the number of examples ensures that the data will be well shuffled.
    During training, the train method typically processes the examples multiple times.
    Calling the tf.data.Dataset.repeat method without any arguments ensures that
    the train method has an infinite supply of (now shuffled) training set examples.

    The train method processes a batch of examples at a time.
    The tf.data.Dataset.batch method creates a batch by concatenating multiple examples.
    This program sets the default batch size to 100,
    meaning that the batch method will concatenate groups of 100 examples. The ideal batch size depends on the problem.
    As a rule of thumb,
    smaller batch sizes usually enable the train method to train the model faster
    at the expense (sometimes) of accuracy.
    """
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """
    An input function for evaluation or prediction

    This function wraps data to return a `tf.data.Dataset`,
    which will be used by TensorFlow as `input_fn` in e.g.
    `classifier.evaluate` and `classifier.predict`
    """
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# The remainder of this file contains a simple example of a csv parser,
#     implemented using a the `Dataset` class.

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

# Modified by Joshua Shields
