"""Utility functions for E2E-VMC."""

import os
import json

import numpy as np
import tensorflow as tf


def count_parameters():
  """Counts all trainable parameters in the tf.DEFAULT_GRAPH."""
  num_params = np.sum(
      [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
  return num_params

def save_model_config(config: dict, run_dir, name):
  """Saves a model config (dict) into the run directory in JSON format."""
  fn = "%s.json" % (name, )
  with open(os.path.join(run_dir, fn), 'w') as f:
    json.dump(config, f, indent=2, sort_keys=True)

def load_model_config(run_dir, name):
  """Loads a model config from a JSON file and returns a config dict."""
  fn = "%s.json" % (name, )
  with open(os.path.join(run_dir, fn), 'r') as f:
    config = json.load(f)
  return config
