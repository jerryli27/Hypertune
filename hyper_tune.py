"""
This file contains functions to help automatically tune hyper-parameters.
Example usage:
# Assume that you would like to tune test_controller.py which has two hyper-parameters x and y.
# And assume that test_controller.py has a function called `hyper_tune()`, which will train the model and
# return a score representing the performance of the current set of hyper-parameters defined in the flags.
python hyper_tune.py --file_name=test_controller --params_and_constraints="[('x',-5,5),('y',-5,5),]"
"""

import ast
import errno
import importlib
import os
import time
from os.path import dirname

import tensorflow as tf
from bayes_opt import BayesianOptimization  # pip install bayesian-optimization
from typing import Union

FLAGS = tf.flags.FLAGS


def _define_flags():
  tf.app.flags.DEFINE_string('file_name', '',
                             'The base file name of the program for which you would like to tune hyperparameters. ')
  tf.app.flags.DEFINE_string('params_and_constraints', '',
                             'The parameters and their constraints represented as a python string representation of '
                             'a list. The list items have format (param_name, lower_lim, upper_lim)')
  tf.app.flags.DEFINE_string('constant_flags', '',
                             'The flags with constant value represented as a python string representation of '
                             'a list. The list items have format (param_name, constant_value)')

  tf.app.flags.DEFINE_integer('init_points', 5,
                              'Number of randomly chosen points to sample the target function initially.')
  tf.app.flags.DEFINE_integer('n_iter', 15,
                              'Total number of times the process is to repeated. ')
  tf.app.flags.DEFINE_string('output_file_name', '',
                             'The file name to store the tune result.')


def _to_param_bound_dict(params):
  ret = dict()
  for param in params:
    if len(param) != 3:
      raise TypeError('Incorrect format for `params_and_constraints`')
    param, lower, upper = param
    if not isinstance(param, str) \
        or not (isinstance(lower, int) or isinstance(lower, float)) \
        or not (isinstance(upper, int) or isinstance(upper, float)):
      raise TypeError('Incorrect format for `params_and_constraints`')
    if lower >= upper:
      raise Exception('For %s, lower_lim %f is greater than upper limit %f!' % (param, float(lower), float(upper)))
    ret[param] = (lower, upper)
  return ret


def touch_folder(file_path):
  # type: (Union[str,unicode]) -> None
  """Create a folder along with its parent folders recursively if they do not exist."""
  # Taken from https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist .
  if not file_path.endswith('/'):
    file_path = file_path + "/"
  dn = dirname(file_path)
  if dn != '':
    try:
      os.makedirs(dn)
    except OSError as e:
      if e.errno != errno.EEXIST:
        raise


def run_python(controller_module, param_val_dict, constant_flags):
  """
  Given the python file name to run and its arguments, return the result as a string.
  :param controller_module: the python module to tune
  :param param_val_dict: a dictionary containing argument names and values
  :return: result as a string.
  """
  for param, val in param_val_dict.iteritems():
    setattr(FLAGS, param, val)  # Set tensorflow flags.
  for param, val in constant_flags:
    setattr(FLAGS, param, val)

  ret = controller_module.hyper_tune()
  return ret


def main(file_name, params_and_constraints, init_points, n_iter, output_file_name, constant_flags_str=''):
  start_time = time.time()
  controller_module = importlib.import_module(file_name)
  # Because we've accessed the FLAGS, the __parsed is set to true, which disallows more flags to be added. We would
  # like to override that to define new flags.
  FLAGS.__dict__['__parsed'] = False
  controller_module.define_flags()

  param_bound_dict = _to_param_bound_dict(ast.literal_eval(params_and_constraints))
  if constant_flags_str:
    constant_flags = ast.literal_eval(constant_flags_str)
  else:
    constant_flags = tuple()
  bo = BayesianOptimization(lambda **kw: run_python(controller_module, kw, constant_flags),
                            param_bound_dict)

  bo.maximize(init_points=init_points, n_iter=n_iter)
  end_time = time.time()

  print('Finished tuning! It took %s seconds. The result is as follows: %s'
        % (str(end_time - start_time), str(bo.res['max'])))
  if output_file_name:
    touch_folder(output_file_name)
    bo.points_to_csv(output_file_name)

  return bo.res['max']


if __name__ == '__main__':
  _define_flags()
  if not FLAGS.file_name:
    raise IOError('Please input a file name (no extension)! Example: python hyper_tune.py --file_name="test"')
  if not os.path.exists(FLAGS.file_name + '.py'):
    raise IOError('File %s does not exist!' % FLAGS.file_name)
  if not FLAGS.params_and_constraints:
    raise AssertionError('You must input the parameters you are trying to tune.')

  main(FLAGS.file_name, FLAGS.params_and_constraints, FLAGS.init_points, FLAGS.n_iter, FLAGS.output_file_name,
       constant_flags_str=FLAGS.constant_flags)
