#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""

https://github.com/tensorflow/models/tree/master/samples/core/get_started

An Example of a DNNClassifier (a Deep Neural Network estimator) for the Iris dataset.

General steps:
Import and parse the data sets.
Create feature columns to describe the data.
Select the type of model
Train the model.
Evaluate the model's effectiveness.
Let the trained model make predictions.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

import iris_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    # Feature columns describe how to use the input.
    # For more information see: https://www.tensorflow.org/get_started/feature_columns
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    """
    my_feature_columns = [
        tf.feature_column.numeric_column(key='SepalLength'),
        tf.feature_column.numeric_column(key='SepalWidth'),
        tf.feature_column.numeric_column(key='PetalLength'),
        tf.feature_column.numeric_column(key='PetalWidth')
    ]
    """

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    """
    The ideal number of hidden layers and neurons depends on the problem and the data set.
    Like many aspects of machine learning,
    picking the ideal shape of the neural network requires some mixture of knowledge and experimentation.
    As a rule of thumb,
    increasing the number of hidden layers and neurons typically creates a more powerful model,
    which requires more data to train effectively.
    """
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each
        hidden_units=[10, 10],
        # The model must choose between 3 classes (the 3 species labels)
        n_classes=3)

    # Train the Model.
    """
    Increasing steps increases the amount of time the model will train.
    Counter-intuitively, training a model longer does not guarantee a better model.
    Choosing the right number of steps usually requires both experience and experimentation.

    IDEA: Couldn't Q-learning be used to select the number of steps?
    https://en.wikipedia.org/wiki/Q-learning
    """
    classifier.train(
        input_fn=lambda: iris_data.train_input_fn(train_x, train_y,
                                                  args.batch_size),
        steps=args.train_steps)

    # Evaluate the model's accuracy based on test data with known labels
    eval_result = classifier.evaluate(
        input_fn=lambda: iris_data.eval_input_fn(test_x, test_y,
                                                 args.batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate label predictions from the model based on input features

    # correct species label for the 3 examples,
    # in a real scenario we would not likely have this,
    # but it provides final validation that the prediction step worked
    expected = ['Setosa', 'Versicolor', 'Virginica']

    # 3 examples to predict the label (species) of based on 4 features each
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    # `labels` being passed as None this time because we are trying to predict the labels,
    # instead of evaluate as done above
    predictions = classifier.predict(
        input_fn=lambda: iris_data.eval_input_fn(predict_x,
                                                 labels=None,
                                                 batch_size=args.batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(iris_data.SPECIES[class_id],
                              100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

# Modified by Joshua Shields
