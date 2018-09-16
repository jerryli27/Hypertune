"""Helper class that has a basic structure of a Reinforcement Learning Controller."""
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
MAX_X_VAL = 2.0


def define_flags():
  tf.app.flags.DEFINE_float('x', 1.0,
                            'Test arg x')
  tf.app.flags.DEFINE_float('y', 1.0,
                            'Test argument y')


def expression(x):
  """This is the mock performance of the model with respect to hyper-parameter x."""
  return np.exp(-(x - 2) ** 2)  # + np.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)


def hyper_tune():
  x = FLAGS.x
  ret = expression(x)
  print('Performance for %.3f is %.3f' %(x, ret))


if __name__ == '__main__':
  define_flags()
  print hyper_tune()
