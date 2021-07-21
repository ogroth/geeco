"""Utility functions for working with tfrecords."""

import numpy as np
import tensorflow as tf


def int_scalar_to_feature(value):
  feat = tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
  return feat

def float_scalar_to_feature(value):
  feat = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
  return feat

def str_scalar_to_feature(value):
  feat = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[value.encode('utf-8')]))
  return feat

def int_list_to_feature(value):
  feat = tf.train.Feature(int64_list=tf.train.Int64List(value=value))
  return feat

def float_list_to_feature(value):
  feat = tf.train.Feature(float_list=tf.train.FloatList(value=value))
  return feat

def str_list_to_feature(value):
  feat = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[v.encode('utf-8') for v in value]))
  return feat

def int_nparray_to_feature(value):
  feat = tf.train.Feature(int64_list=tf.train.Int64List(value=value.flatten()))
  return feat

def float_nparray_to_feature(value):
  feat = tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))
  return feat


def convert_to_feature(value):
  """
  Converts a value to tf.train.Feature(value).
  Infers the type and shape of value before conversion.
  Value must either be a Python primitive datatype or np.ndarray.
  """
  t_feat = type(value)  # infer top-level type
  # supported scalar types
  if t_feat == int:
    feat = int_scalar_to_feature(value)
  elif t_feat == float:
    feat = float_scalar_to_feature(value)
  elif t_feat == str:
    feat = str_scalar_to_feature(value)
  # supported list types
  elif t_feat == list:
    t_elem = type(value[0])  # infer element type of list
    if t_elem == int:
      feat = int_list_to_feature(value)
    elif t_elem == float:
      feat = float_list_to_feature(value)
    elif t_elem == str:
      feat = str_list_to_feature(value)
    else:
      err_msg = "Unsupported conversion of list type %s to tf.train.Feature!" % (t_elem, )
      raise TypeError(err_msg)
  # supported numpy array types
  elif t_feat == np.ndarray:  # np array, flatten
    t_elem = value.dtype  # infer np.dtype
    if t_elem == np.int32 or t_elem == np.int64:
      feat = int_nparray_to_feature(value)
    elif t_elem == np.float32 or t_elem == np.float64 or t_elem == np.uint8:  # matrices and images!
      feat = float_nparray_to_feature(value)
    else:
      err_msg = "Unsupported conversion of array type %s to tf.train.Feature!" % (t_elem, )
      raise TypeError(err_msg)
  else:
    err_msg = "Unsupported conversion of type %s to tf.train.Feature!" % (t_feat, )
    raise TypeError(err_msg)
  return feat
