"""High-level predictor API for E2E-VMC."""

import os
import pprint

import numpy as np
import tensorflow as tf

from .graph import e2e_vmc, goal_e2evmc
# from .estimator import e2evmc_model_fn
from .utils import load_model_config
from .params import create_e2evmc_config


# ---------- constants ----------

TOL_FRAME_RANGE = 1e-6  # tolerance for value range of fed frames


# ---------- helper_fn ----------

# @DEPRECATED: was used with Estimator.predict()
def _input_fn(frames):
  features = {
      'rgb' : [],
      'jnt_state' : [],
  }
  # aggregate frame dicts
  for frame in frames:
    for k in features.keys():
      features[k].append(frame[k])
  for k, v in features.items():
    print(k, [arr.shape for arr in v])
  # transform into feature tensor
  for k in features:
    features[k] = tf.convert_to_tensor(np.array(features[k]), dtype=tf.float32)
  dataset = tf.data.Dataset.from_tensors(features)
  return dataset


# ---------- predictors ----------

class GoalE2EVMCPredictor(object):
  """High-level API to run goal-conditioned E2EVMC."""

  def __init__(self, model_dir, checkpoint_name=None, memcap=0.8):
    """
    GoalE2EVMCPredictor
    :param model_dir:
    :param checkpoint_name:
    :param memcap:
    """
    self._model_dir = model_dir
    # load model config
    cfg = load_model_config(model_dir, 'e2evmc_config')
    cfg['batch_size'] = 1  # fix to one prediction at a time
    cfg = create_e2evmc_config(cfg)
    self._cfg = cfg
    # create input placeholders
    self._input_ph = {
        'rgb_frames' : tf.placeholder(
            dtype=tf.float32,
            shape=[1, cfg.window_size, cfg.img_height, cfg.img_width, cfg.img_channels],
            name='ph_rgb_frames'),
        'jnt_states' : tf.placeholder(
            dtype=tf.float32,
            shape=[1, cfg.window_size, cfg.dim_jnt_state],
            name='ph_jnt_states'),
        'tgt_frame' : tf.placeholder(
            dtype=tf.float32,
            shape=[1, cfg.img_height, cfg.img_width, cfg.img_channels],
            name='ph_tgt_frame'),
        'reset' : tf.placeholder(dtype=tf.bool, shape=[], name='ph_reset'),
    }
    # create e2evmc graph
    self._net, self._eps = goal_e2evmc(
        self._input_ph['rgb_frames'], self._input_ph['jnt_states'], self._input_ph['tgt_frame'],
        self._input_ph['reset'],
        params=self._cfg)
    print(">>> Instantiated GoalE2EVMC:")  # DEBUG
    pprint.pprint(self._eps)
    # start interactive session
    self._sess = tf.InteractiveSession()
    # load checkpoint
    self._sess.run(tf.global_variables_initializer())
    print(">>> Initialized model graph.")
    var_list = [v for v in tf.global_variables() if v.name != 'GoalVMC/LSTMDecoder/lstm_memory:0']  # TODO: remove hard-coded name!
    saver = tf.train.Saver(var_list)
    if not checkpoint_name:
      ckpt_path = tf.train.latest_checkpoint(model_dir)
    else:
      ckpt_path = os.path.join(model_dir, checkpoint_name)
    saver.restore(self._sess, save_path=ckpt_path)
    print(">>> Restored model parameters from %s" % (ckpt_path, ))  # DEBUG
    pprint.pprint(var_list)  # DEBUG
    # frame buffer
    self._frame_buffer = []
    self._buffer_size = cfg.window_size
    self._target_frame = None

  @property
  def sess(self):
    """Expose the underlying tensorflow session."""
    return self._sess

  @property
  def cfg(self):
    """Expose the predictor config."""
    return self._cfg

  def _prepare_frame_buffer(self):
    """Prepares the frame buffer for tf.Session feeding."""
    features = {
        'rgb' : [],
        'jnt_state' : [],
    }
    # aggregate frame dicts
    for frame in self._frame_buffer:
      for k in features.keys():
        features[k].append(frame[k])
    # transform into batched feature tensor
    for k in features:
      features[k] = np.array(features[k])
      features[k] = np.expand_dims(features[k], axis=0)
    return features

  def _feed_frame(self, rgb_frame, jnt_state):
    """Feed frame (rgb, jnt_state) and maintain frame buffer."""
    # check shape and range of fed frame
    received_shape = rgb_frame.shape
    expected_shape = (self._cfg.img_height, self._cfg.img_width, self._cfg.img_channels)
    err_msg = "Fed frame has wrong dimensions! Expected %s, got %s!" % (expected_shape, received_shape)
    assert received_shape == expected_shape, err_msg
    received_range = (np.amin(rgb_frame), np.amax(rgb_frame))
    expected_range = (0 - TOL_FRAME_RANGE, 1 + TOL_FRAME_RANGE)
    err_msg = "Fed frame exceeds range! Expected %s, got %s!" % (expected_range, received_range)
    assert expected_range[0] <= received_range[0] <= expected_range[1] \
      and expected_range[0] <= received_range[1] <= expected_range[1], err_msg
    # build frame and feed
    frame = {
        'rgb' : rgb_frame,
        'jnt_state' : jnt_state,
    }
    self._frame_buffer.append(frame)
    if len(self._frame_buffer) > self._buffer_size:
      self._frame_buffer.pop(0)

  def _predict_command(self, reset):
    """Predict next command based on current frame buffer."""
    if self._cfg.control_mode == 'cartesian':
      fetches = {
          'cmd_ee' : self._eps['pred_cmd_ee'],
          'cmd_grp' : self._eps['logits_cmd_grp'],
          'pos_ee' : self._eps['pred_aux_ee'],
          'pos_obj' : self._eps['pred_aux_obj'],
      }
    elif self._cfg.control_mode == 'velocity':
      fetches = {
          'cmd_vel' : self._eps['pred_cmd_vel'],
          'cmd_ee' : self._eps['pred_cmd_ee'],
          'cmd_grp' : self._eps['pred_cmd_grp'],
          'pos_ee' : self._eps['pred_aux_ee'],
          'pos_obj' : self._eps['pred_aux_obj'],
      }
    # add debug output, if supported by the model
    if self._cfg.proc_obs == 'dynimg':
      fetches['dynbuff'] = self._eps['dynbuff']
    if self._cfg.proc_tgt == 'dyndiff':
      fetches['dyndiff'] = self._eps['dyndiff']
    # prepare & batch features
    features = self._prepare_frame_buffer()
    feed_dict = {
        self._input_ph['rgb_frames'] : features['rgb'],
        self._input_ph['jnt_states'] : features['jnt_state'],
        self._input_ph['tgt_frame'] : np.expand_dims(self._target_frame, axis=0),
        self._input_ph['reset'] : np.array(reset, dtype=np.bool),
    }
    predictions = self._sess.run(fetches, feed_dict)
    # unbatch predictions
    for k in predictions.keys():
      predictions[k] = np.squeeze(predictions[k])
    # re-map discrete gripper command predictions
    if self._cfg.control_mode == 'cartesian':
      cmd_grp = predictions['cmd_grp']  # logits!
      cmd_grp = np.argmax(cmd_grp)  # get class of gripper action
      cmd_grp = cmd_grp.reshape((1, ))
      cmd_grp -= 1 # shift class indices by one
      cmd_grp = cmd_grp.astype(np.float32)
      predictions['cmd_grp'] = cmd_grp
    return predictions

  def predict(self, rgb_frame, jnt_state):
    """Feeds the given frame (and pads the frame buffer) and returns the predictions."""
    reset = (len(self._frame_buffer) == 0)  # send reset signal, if frame buffer was empty
    self._feed_frame(rgb_frame, jnt_state)
    # pad frame buffer, if necessary
    while len(self._frame_buffer) < self._buffer_size:
      self._feed_frame(rgb_frame, jnt_state)
    predictions = self._predict_command(reset)
    return predictions

  def reset(self):
    """Resets the predictor by clearing the frame buffer."""
    self._frame_buffer.clear()

  def set_goal(self, tgt_frame):
    """Sets the target frame for the predictor."""
    self._target_frame = np.copy(tgt_frame[:, :, :self._cfg.img_channels])



class E2EVMCPredictor(object):
  """High-level API to run E2E VMC."""

  def __init__(self, model_dir, checkpoint_name=None, memcap=0.8):
    """
    E2EVMCPredictor
    :param model_dir:
    :param checkpoint_name:
    :param memcap:
    """
    self._model_dir = model_dir
    # load model config
    cfg = load_model_config(model_dir, 'e2evmc_config')
    cfg['batch_size'] = 1  # fix to one prediction at a time
    cfg = create_e2evmc_config(cfg)
    self._cfg = cfg
    # create input placeholders
    self._input_ph = {
        'rgb_frames' : tf.placeholder(
            dtype=tf.float32,
            shape=[1, cfg.window_size, cfg.img_height, cfg.img_width, cfg.img_channels],
            name='ph_rgb_frames'),
        'jnt_states' : tf.placeholder(
            dtype=tf.float32,
            shape=[1, cfg.window_size, cfg.dim_jnt_state],
            name='ph_jnt_states'),
        'reset' : tf.placeholder(dtype=tf.bool, shape=[], name='ph_reset'),
    }
    # create e2evmc graph
    self._net, self._eps = e2e_vmc(
        self._input_ph['rgb_frames'], self._input_ph['jnt_states'], self._input_ph['reset'],
        params=self._cfg)
    print(">>> Instantiated E2EVMC:")  # DEBUG
    pprint.pprint(self._eps)
    # start interactive session
    self._sess = tf.InteractiveSession()
    # load checkpoint
    self._sess.run(tf.global_variables_initializer())
    print(">>> Initialized model graph.")
    var_list = [v for v in tf.global_variables() if v.name != 'VMC/LSTMDecoder/lstm_memory:0']  # TODO: remove hard-coded name!
    saver = tf.train.Saver(var_list)
    if not checkpoint_name:
      ckpt_path = tf.train.latest_checkpoint(model_dir)
    else:
      ckpt_path = os.path.join(model_dir, checkpoint_name)
    saver.restore(self._sess, save_path=ckpt_path)
    print(">>> Restored model parameters from %s" % (ckpt_path, ))  # DEBUG
    pprint.pprint(var_list)  # DEBUG
    # frame buffer
    self._frame_buffer = []
    self._buffer_size = cfg.window_size

    # --- DEPRECATED: Estimator.predict(); kept for reference
    # # input_fn
    # self._input_fn = lambda: _input_fn(self._frame_buffer)
    # # define run config
    # gpu_options = tf.GPUOptions(allow_growth=True, per_process_gpu_memory_fraction=memcap)
    # sess_config = tf.ConfigProto(gpu_options=gpu_options)
    # run_config = tf.estimator.RunConfig(session_config=sess_config)
    # # set up estimator in predict mode (load checkpoint_name, if given; else: latest)
    # estimator_params = {
    #     'e2evmc_config' : cfg,
    #     'debug' : False,
    # }
    # self._estimator = tf.estimator.Estimator(
    #     model_fn=e2evmc_model_fn,
    #     model_dir=model_dir,
    #     config=run_config,
    #     params=estimator_params)

  @property
  def sess(self):
    """Expose the underlying tensorflow session."""
    return self._sess

  @property
  def cfg(self):
    """Expose the predictor config."""
    return self._cfg

  def _prepare_frame_buffer(self):
    """Prepares the frame buffer for tf.Session feeding."""
    features = {
        'rgb' : [],
        'jnt_state' : [],
    }
    # aggregate frame dicts
    for frame in self._frame_buffer:
      for k in features.keys():
        features[k].append(frame[k])
    # transform into batched feature tensor
    for k in features:
      features[k] = np.array(features[k])
      features[k] = np.expand_dims(features[k], axis=0)
    return features

  def _feed_frame(self, rgb_frame, jnt_state):
    """Feed frame (rgb, jnt_state) and maintain frame buffer."""
    # check shape and range of fed frame
    received_shape = rgb_frame.shape
    expected_shape = (self._cfg.img_height, self._cfg.img_width, self._cfg.img_channels)
    err_msg = "Fed frame has wrong dimensions! Expected %s, got %s!" % (expected_shape, received_shape)
    assert received_shape == expected_shape, err_msg
    received_range = (np.amin(rgb_frame), np.amax(rgb_frame))
    expected_range = (0 - TOL_FRAME_RANGE, 1 + TOL_FRAME_RANGE)
    err_msg = "Fed frame exceeds range! Expected %s, got %s!" % (expected_range, received_range)
    assert expected_range[0] <= received_range[0] <= expected_range[1] \
      and expected_range[0] <= received_range[1] <= expected_range[1], err_msg
    # build frame and feed
    frame = {
        'rgb' : rgb_frame,
        'jnt_state' : jnt_state,
    }
    self._frame_buffer.append(frame)
    if len(self._frame_buffer) > self._buffer_size:
      self._frame_buffer.pop(0)

  def _predict_command(self, reset):
    """Predict next command based on current frame buffer."""
    if self._cfg.control_mode == 'cartesian':
      fetches = {
          'cmd_ee' : self._eps['pred_cmd_ee'],
          'cmd_grp' : self._eps['logits_cmd_grp'],
          'pos_ee' : self._eps['pred_aux_ee'],
          'pos_obj' : self._eps['pred_aux_obj'],
      }
    elif self._cfg.control_mode == 'velocity':
      fetches = {
          'cmd_vel' : self._eps['pred_cmd_vel'],
          'cmd_ee' : self._eps['pred_cmd_ee'],
          'cmd_grp' : self._eps['pred_cmd_grp'],
          'pos_ee' : self._eps['pred_aux_ee'],
          'pos_obj' : self._eps['pred_aux_obj'],
      }
    # prepare & batch features
    features = self._prepare_frame_buffer()
    feed_dict = {
        self._input_ph['rgb_frames'] : features['rgb'],
        self._input_ph['jnt_states'] : features['jnt_state'],
        self._input_ph['reset'] : np.array(reset, dtype=np.bool),
    }
    predictions = self._sess.run(fetches, feed_dict)
    # unbatch predictions
    for k in predictions.keys():
      predictions[k] = np.squeeze(predictions[k])
    # re-map discrete gripper command predictions
    if self._cfg.control_mode == 'cartesian':
      cmd_grp = predictions['cmd_grp']  # logits!
      cmd_grp = np.argmax(cmd_grp)  # get class of gripper action
      cmd_grp = cmd_grp.reshape((1, ))
      cmd_grp -= 1 # shift class indices by one
      cmd_grp = cmd_grp.astype(np.float32)
      predictions['cmd_grp'] = cmd_grp
    return predictions

  def predict(self, rgb_frame, jnt_state):
    """Feeds the given frame (and pads the frame buffer) and returns the predictions."""
    reset = (len(self._frame_buffer) == 0)  # send reset signal, if frame buffer was empty
    self._feed_frame(rgb_frame, jnt_state)
    # pad frame buffer, if necessary
    while len(self._frame_buffer) < self._buffer_size:
      self._feed_frame(rgb_frame, jnt_state)
    predictions = self._predict_command(reset)
    return predictions

  def reset(self):
    """Resets the predictor by clearing the frame buffer."""
    self._frame_buffer.clear()
