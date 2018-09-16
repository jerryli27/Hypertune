"""Helper class that has a basic structure of a Reinforcement Learning Controller."""
import tensorflow as tf
import numpy as np

FLAGS = tf.flags.FLAGS
MAX_X_VAL = 2.0

def define_flags():
    tf.app.flags.DEFINE_float('x', 1.0,
                               'Test arg x')
    tf.app.flags.DEFINE_float('y', 1.0,
                               'Test argument y')

def expression(x):
    return np.exp(-(x - 2)**2) + np.exp(-(x - 6)**2/10) + 1/ (x**2 + 1)

def hyper_tune():
    x = FLAGS.x
    return expression(x)


if __name__ == '__main__':
    define_flags()
    print hyper_tune()