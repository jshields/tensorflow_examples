#!/usr/bin/env python3
import tensorflow as tf
import os

if __name__ == '__main__':
    # suppress the warnings (for me, about unsupported CPU features)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # validate that tensorflow is functioning
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    greeting = sess.run(hello).decode()
    print(greeting)

# Modified by Joshua Shields
