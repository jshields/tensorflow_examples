# Tensorflow Examples

Trying out TensorFlow

**Disclaimer:**
This Source and Work may be considered Derivative Works of https://github.com/tensorflow/models.
Attribution notices list the original copyright, authors and contributors,
and will not reflect the changes or additional files the owner of this repository may have added.
Some or all files may be be modified from the original source.
Modifications may or may not consist of substantial changes, including but not limited to changing the file path or file contents.
Some or all files may or may not be from the original source, and not all files from the original source may be present in this repository.
A good faith effort has been made to mark all relevant Source files with a modification notice in order to satisfy the LICENSE,
as well as to meet all other requirements of the LICENSE.


Initial ideas:
* flower classification
* digit classification


Some Example Datasets:
* Iris flowers
  * https://en.wikipedia.org/wiki/Iris_flower_data_set: http://archive.ics.uci.edu/ml/datasets/Iris
  * https://www.tensorflow.org/get_started/get_started_for_beginners#the_iris_classification_problem
* hand written digits
  * https://en.wikipedia.org/wiki/MNIST_database: http://yann.lecun.com/exdb/mnist/
  * Optical Recognition of Handwritten Digits Dataset: http://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits


## Usage:

Note to self if checking out on a new machine:

    sudo chmod +x filemarker.sh
    cp filemarker.sh .git/hooks/pre-commit


Install:

    virtualenv --python=python3 ~/virtualenvs/ml
    source ~/virtualenvs/ml/bin/activate
    pip install -r requirements.txt

Validate Tensorflow installation:

    python validate_tensorflow.py

Local optimization warning, may want to consider using GPU support,
building a version compatible with the CPU,
or using [docker](https://hub.docker.com/r/tensorflow/tensorflow/):

> Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA

https://stackoverflow.com/q/47068709/3538313
