"""
stub
"""

import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from .params import E2E_VMC_DEFAULT_CONFIG


# region: helper_fn

def _weight_regularizer(params=E2E_VMC_DEFAULT_CONFIG):
  regularizer = l2_regularizer(scale=params.l2_regularizer)
  return regularizer

def _H(t):
  """Computes the t-th harmonic number."""
  H_t = tf.cond(
      tf.equal(t, 0),
      lambda: tf.constant(0.0),
      lambda: tf.reduce_sum(1.0 / tf.range(1, t+1, dtype=tf.float32)))
  return H_t

def _alpha(t, T):
  """Computes the dynimg coefficient for the t-th tensor in a sequence of length T."""
  a_t = 2 * (T - t + 1) - (T + 1) * (_H(T) - _H(t-1))
  return a_t

def dynimg(rgb_frames, params=E2E_VMC_DEFAULT_CONFIG, scope='DynImg'):
  """
  Computes the (normalized) dynamic image from a sequence of RGB frames.

  :param rgb_frames: [N, K, H, W, C], tf.float32 in [0.0 .. 1.0]

  :return dynimg: [N, H, W, C], tf.float32 in [0.0 .. 1.0]
  """
  seq_shape = rgb_frames.get_shape().as_list()
  K = seq_shape[1]
  # compute dynamic image
  frame_idx = tf.range(1, K+1, dtype=tf.float32)
  frame_weights = tf.map_fn(lambda t: _alpha(t, T=K), frame_idx)
  frame_weights = tf.reshape(frame_weights, [1, K, 1, 1, 1])  # explicit reshape to multiply along sequence axis
  weighted_frames = frame_weights * rgb_frames
  dyn_images = tf.reduce_sum(weighted_frames, axis=1)  # sum along frame sequences
  # normalize dynamic image
  min_batched = tf.reduce_min(dyn_images, axis=[1, 2, 3])  # compute min per dyn_img in batch
  max_batched = tf.reduce_max(dyn_images, axis=[1, 2, 3])  # compute max per dyn_img in batch
  img_range = max_batched - min_batched + 1e-6  # add small epsilon to avoid division by zero
  # reshape
  min_batched = tf.reshape(min_batched, [-1, 1, 1, 1])
  max_batched = tf.reshape(max_batched, [-1, 1, 1, 1])
  img_range = tf.reshape(img_range, [-1, 1, 1, 1])
  norm_dyn_images = (dyn_images - min_batched) / img_range
  return norm_dyn_images

# endregion

# region: frame encoder

def conv_encoder(
    rgb_frame, dim_out=256, params=E2E_VMC_DEFAULT_CONFIG, scope='ConvEncoder'):
  """
  Convolves the RGB frame into a vector representation.

  :param rgb_frame: [N, img_height, img_width, img_channels]
  :param params: E2EVMCConfig

  :return net: [N, 2 * 2 * 256] (encoding vector of the input frame)
  :return endpoints: dict of relevant endpoint nodes
  """
  regularizer = _weight_regularizer(params)
  with tf.variable_scope(scope, regularizer=regularizer, reuse=tf.AUTO_REUSE):
    endpoints = {}
    # --- convolve RGB(-D) frame
    net = tf.layers.conv2d(  # -> 256, 256, 32
        inputs=rgb_frame,
        filters=32, kernel_size=3, strides=1, padding='SAME',
        activation=tf.nn.relu,
        name='conv1')
    net = tf.layers.conv2d(  # -> 128, 128, 48
        inputs=net,
        filters=48, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv2')
    net = tf.layers.conv2d(  # -> 64, 64, 64
        inputs=net,
        filters=64, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv3')
    net = tf.layers.conv2d(  # -> 32, 32, 128
        inputs=net,
        filters=128, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv4')
    net = tf.layers.conv2d(  # -> 16, 16, 192
        inputs=net,
        filters=192, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv5')
    net = tf.layers.conv2d(  # -> 8, 8, 256
        inputs=net,
        filters=256, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv6')
    net = tf.layers.conv2d(  # -> 4, 4, 256
        inputs=net,
        filters=256, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv7')
    net = tf.layers.conv2d(  # 2, 2, 256
        inputs=net,
        filters=dim_out, kernel_size=3, strides=2, padding='SAME',
        activation=tf.nn.relu,
        name='conv8')
    endpoints['conv8'] = net
    return net, endpoints

# endregion

# region: state representation

def state_concatenation(
    frame_feat, jnt_state, params=E2E_VMC_DEFAULT_CONFIG, scope='StateConcat'):
  """
  Concatenate the joint state to the frame features (variable-free operations).

  :param frame_feat: [N, 2, 2, 256]
  :param jnt_state: [N, dim_jnt_state]
  :param params: E2EVMCConfig

  :return net: [N, 2 * 2 * (256 + dim_jnt_state)] (encoding vector of the input frame)
  :return endpoints: dict of relevant endpoint nodes
  """
  regularizer = _weight_regularizer(params)
  with tf.variable_scope(scope, regularizer=regularizer, reuse=tf.AUTO_REUSE):
    endpoints = {}
    state = tf.reshape(jnt_state, [-1, 1, 1, params.dim_jnt_state])
    state = tf.tile(state, [1, 2, 2, 1])  # TODO: infer spatial tiling from last conv tensor
    net = tf.concat([frame_feat, state], axis=-1, name='concat_jnt_state')  # concat along channels
    net = tf.layers.flatten(inputs=net, name='flatten_enc')
    endpoints['flat_state'] = net
    # TODO: further processing? convolutions? skips?
    return net, endpoints

def representation_concatenation(
    obs_feat, tgt_feat, jnt_state, params=E2E_VMC_DEFAULT_CONFIG, scope='RepresentationConcat'):
  """
  Concatenate the joint state to the frame features (variable-free operations).

  :param obs_feat: [N, 2, 2, 256], feature encoding of the currently observed frame
  :param tgt_feat: [N, 2, 2, 256], feature encoding of the target frame
  :param jnt_state: [N, dim_jnt_state], joint state (proprioception)
  :param params: E2EVMCConfig

  :return net: [N, 2 * 2 * (256 + 256 + dim_jnt_state)] (encoding vector of the input frame)
  :return endpoints: dict of relevant endpoint nodes
  """
  regularizer = _weight_regularizer(params)
  with tf.variable_scope(scope, regularizer=regularizer, reuse=tf.AUTO_REUSE):
    endpoints = {}
    state = tf.reshape(jnt_state, [-1, 1, 1, params.dim_jnt_state])
    state = tf.tile(state, [1, 2, 2, 1])  # TODO: infer spatial tiling from last conv tensor
    net = tf.concat([obs_feat, state, tgt_feat], axis=-1, name='concat_jnt_state')  # concat along channels
    net = tf.layers.flatten(inputs=net, name='flatten_enc')
    endpoints['flat_representation'] = net
    return net, endpoints

def representation_concatenation_v2(
    obs_feat, dyn_feat, jnt_state, tgt_feat,
    params=E2E_VMC_DEFAULT_CONFIG, scope='RepresentationConcat'):
  """
  Concatenate the parts of the state (observation, dynamics, proprioception, target).

  :param obs_feat: [N, 2, 2, 256], feature encoding of the currently observed frame
  :param dyn_feat: [N, 2, 2, 256], feature encoding of the dynamics of the frame buffer
  :param jnt_state: [N, dim_jnt_state], joint state (proprioception)
  :param tgt_feat: [N, 2, 2, 256], feature encoding of the target frame
  :param params: E2EVMCConfig

  :return net: [N, 2 * 2 * (256 + 256 + dim_jnt_state + 256)] (state tensor)
  :return endpoints: dict of relevant endpoint nodes
  """
  regularizer = _weight_regularizer(params)
  with tf.variable_scope(scope, regularizer=regularizer, reuse=tf.AUTO_REUSE):
    endpoints = {}
    jnt_state = tf.reshape(jnt_state, [-1, 1, 1, params.dim_jnt_state])
    jnt_state = tf.tile(jnt_state, [1, 2, 2, 1])  # TODO: infer spatial tiling from last conv tensor
    net = tf.concat([obs_feat, dyn_feat, jnt_state, tgt_feat], axis=-1, name='concat_state')  # concat along channels
    net = tf.layers.flatten(inputs=net, name='flatten_state_enc')
    endpoints['flat_representation'] = net
    return net, endpoints

# endregion

# region: command decoder

def lstm_decoder(
    feat_list, reset, params=E2E_VMC_DEFAULT_CONFIG, scope='LSTMDecoder'):
  """
  Feeds a list of feature representations into an LSTM and decodes the last output
  vector into joint velocities and EE and gripper commands.
  Also decodes auxiliary predictions of EE and object position.

  :param feat_list: K x [N, feat_dim]
  :param reset: [], tf.bool; indicates whether the LSTM is reset because a new sequence starts
  :param params: E2EVMCConfig

  :return net: [N, dim_h_fc] (FC layer before prediction heads)
  :return endpoints: dict of relevant endpoint nodes
  """
  batch_size = params.batch_size
  regularizer = _weight_regularizer(params)
  with tf.variable_scope(scope, regularizer=regularizer, reuse=tf.AUTO_REUSE):
    endpoints = {}
    # TODO: replace vanilla LSTM with optimized tf.contrib.cudnn_rnn.CudnnLSTM
    lstm = tf.nn.rnn_cell.LSTMCell(num_units=params.dim_h_lstm, state_is_tuple=False)
    lstm_init = tf.zeros([batch_size, 2 * params.dim_h_lstm], dtype=tf.float32)  # 2 x dim_h_lstm for hidden state and output
    lstm_memory = tf.Variable(initial_value=lstm_init, trainable=False, name='lstm_memory')  # TODO: exclude from saved vars!
    lstm_state = tf.cond(reset, lambda: lstm.zero_state(batch_size, tf.float32), lambda: lstm_memory)
    outputs = []
    # --- feed all features into LSTM and collect outputs
    for feat in feat_list:
      lstm_out, lstm_state = lstm(inputs=feat, state=lstm_state)
      outputs.append(lstm_out)
    tf.assign(lstm_memory, lstm_state)  # save last LSTM state
    # TODO: refactor prediction heads into separate graph_fn
    # --- pass last output into fully connected layer
    net = tf.layers.dense(
        inputs=outputs[-1], units=params.dim_h_fc, activation=tf.nn.relu, name='fc1')
    endpoints['fc1'] = net
    # --- attach target decoder heads
    if params.control_mode == 'cartesian':
      pred_cmd_ee = tf.layers.dense(  # command for delta x-y-z of EE
          inputs=net, units=3, activation=None, name='pred_cmd_ee')
      endpoints['pred_cmd_ee'] = pred_cmd_ee
      logits_cmd_grp = tf.layers.dense(  # logits of gripper command: -1 = close; 0 = no-op; 1 = open
          inputs=net, units=params.num_grp_states, activation=None, name='logits_cmd_grp')
      endpoints['logits_cmd_grp'] = logits_cmd_grp
    elif params.control_mode == 'velocity':
      pred_cmd_vel = tf.layers.dense(  # command for absolute target joint velocities
          inputs=net, units=params.dim_jnt_state, activation=None, name='pred_cmd_vel')
      endpoints['pred_cmd_vel'] = pred_cmd_vel
      pred_cmd_ee = tf.layers.dense(  # command for absolute x-y-z of EE
          inputs=net, units=3, activation=None, name='pred_cmd_ee')
      endpoints['pred_cmd_ee'] = pred_cmd_ee
      pred_cmd_grp = tf.layers.dense(  # command for absolute finger positions
          inputs=net, units=params.dim_grp_command, activation=None, name='pred_cmd_grp')
      endpoints['pred_cmd_grp'] = pred_cmd_grp
    else:
      err_msg = "Unknown control mode '%s'" % (params.concat, )
      raise ValueError(err_msg)
    # --- attach auxiliary decoder heads
    pred_aux_ee = tf.layers.dense(  # prediction of current x-y-z of EE
        inputs=net, units=3, activation=None, name='pred_aux_ee')
    endpoints['pred_aux_ee'] = pred_aux_ee
    pred_aux_obj = tf.layers.dense(  # prediction of current x-y-z of object
        inputs=net, units=3, activation=None, name='pred_aux_obj')
    endpoints['pred_aux_obj'] = pred_aux_obj
    return net, endpoints

# TODO: add FC decoder

# endregion

# region: model_fn

def e2e_vmc(
    rgb_frames, jnt_states, reset, params=E2E_VMC_DEFAULT_CONFIG, scope='VMC'):
  """
  Graph_fn of E2E-VMC.

  Baseline model implementation based on:
  @article{DBLP:journals/corr/JamesDJ17,
    author    = {Stephen James and
                Andrew J. Davison and
                Edward Johns},
    title     = {Transferring End-to-End Visuomotor Control from Simulation to Real
                World for a Multi-Stage Task},
    journal   = {CoRR},
    volume    = {abs/1707.02267},
    year      = {2017},
    url       = {http://arxiv.org/abs/1707.02267},
    archivePrefix = {arXiv},
    eprint    = {1707.02267},
    timestamp = {Mon, 13 Aug 2018 16:47:24 +0200},
    biburl    = {https://dblp.org/rec/bib/journals/corr/JamesDJ17},
    bibsource = {dblp computer science bibliography, https://dblp.org}
  }

  :param rgb_frames: [N, window_size, img_height, img_width, img_channels]
  :param jnt_states: [N, window_size, dim_jnt_state]
  :param reset: [], tf.bool; indicates whether a new sequence starts
  :param params: E2EVMCConfig

  :return net: [N, dim_h_fc] (FC of LSTM output, before prediction heads)
  :return endpoints: dict of relevant endpoint nodes
  """
  with tf.variable_scope(scope):
    endpoints = {}
    # --- process frames individually and collect features
    # reshape explicitly to force shape inference before unstack
    rgb_frames = tf.reshape(
        rgb_frames, [-1, params.window_size, params.img_height, params.img_width, params.img_channels])
    jnt_states = tf.reshape(
        jnt_states, [-1, params.window_size, params.dim_jnt_state])
    rgb_frame_list = tf.unstack(rgb_frames, axis=1)
    jnt_state_list = tf.unstack(jnt_states, axis=1)
    feat_list = []
    for rgb_frame, jnt_state in zip(rgb_frame_list, jnt_state_list):
      feat, ep_enc = conv_encoder(rgb_frame, params=params)
      state, ep_state = state_concatenation(feat, jnt_state, params=params)
      feat_list.append(state)
    endpoints.update(ep_enc)
    endpoints.update(ep_state)
    # --- decode feature list with LSTM
    net, ep_dec = lstm_decoder(feat_list, reset, params=params)
    endpoints.update(ep_dec)
    return net, endpoints

def goal_e2evmc(
    rgb_frames, jnt_states, tgt_frame, reset, params=E2E_VMC_DEFAULT_CONFIG, scope='GoalVMC'):
  """
  Graph_fn of goal-driven E2E-VMC.

  Official implementation of:
  @inproceedings{groth2021geeco,
    author={Oliver Groth and Chia-Man Hung and Andrea Vedaldi and Ingmar Posner},
    booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)}, 
    title={Goal-Conditioned End-to-End Visuomotor Control for Versatile Skill Primitives}
  }

  :param rgb_frames: [N, window_size, img_height, img_width, img_channels]
  :param jnt_states: [N, window_size, dim_jnt_state]
  :param tgt_frame: [N, img_height, img_width, img_channels]
  :param reset: [], tf.bool; indicates whether a new sequence starts
  :param params: E2EVMCConfig

  :return net: [N, dim_h_fc] (FC of LSTM output, before prediction heads)
  :return endpoints: dict of relevant endpoint nodes
  """
  with tf.variable_scope(scope):
    endpoints = {}
    # --- reshape explicitly to force shape inference before unstack
    rgb_frames = tf.reshape(
        rgb_frames, [-1, params.window_size, params.img_height, params.img_width, params.img_channels])
    jnt_states = tf.reshape(
        jnt_states, [-1, params.window_size, params.dim_jnt_state])
    rgb_frame_list = tf.unstack(rgb_frames, axis=1)
    jnt_state_list = tf.unstack(jnt_states, axis=1)
    # --- encode target frame
    if params.proc_tgt in ['constant', 'residual']:
      # encode target image with same conv-stack as frame buffer
      tgt_feat, ep_tgt = conv_encoder(tgt_frame, dim_out=params.dim_s_obs, params=params)
    elif params.proc_tgt == 'dyndiff':
      pass  # encoded later as dynamic image against current frame buffer
    else:
      err_msg = "Unknown processing mode for target image: %s!" % (params.proc_tgt, )
      raise ValueError(err_msg)
    # --- build feature list to feed into LSTM
    feat_list = []
    if params.proc_obs == 'sequence':
      for rgb_frame, jnt_state in zip(rgb_frame_list, jnt_state_list):
        feat, ep_enc = conv_encoder(rgb_frame, dim_out=params.dim_s_obs, params=params)
        if params.proc_tgt == 'constant':
          # concatenate target feature to state
          state, ep_state = representation_concatenation(feat, tgt_feat, jnt_state, params=params)
        elif params.proc_tgt == 'residual':
          res_feat = tgt_feat - feat  # compute residual feature between frame and target
          state, ep_state = state_concatenation(res_feat, jnt_state, params=params)
        elif params.proc_tgt == 'dyndiff':  # TODO: restrict to RGB part of observations?
          # compute dynamic image difference with target frame and encode with separate conv-stack
          frame_cur = tf.expand_dims(rgb_frame, axis=1)  # introduce sequence axis to concat along
          frame_tgt = tf.expand_dims(tgt_frame, axis=1)  # introduce sequence axis to concat along
          seq_diff = tf.concat([frame_cur, frame_tgt], axis=1)
          dyn_diff = dynimg(seq_diff)
          endpoints['dyndiff'] = dyn_diff  # register EP to return dyndiff in forward pass
          tgt_feat, ep_tgt = conv_encoder(dyn_diff, dim_out=params.dim_s_diff, params=params, scope='DynDiffEncoder')
          endpoints.update(ep_tgt)
          # concatenate target feature to state
          state, ep_state = representation_concatenation(feat, tgt_feat, jnt_state, params=params)
        else:
          err_msg = "Unknown processing mode for target image: %s!" % (params.proc_tgt, )
          raise ValueError(err_msg)
        feat_list.append(state)
    elif params.proc_obs == 'dynimg':
      rgb_frame = rgb_frame_list[-1]
      jnt_state = jnt_state_list[-1]
      # current RGB feature
      feat, ep_enc = conv_encoder(rgb_frame, dim_out=params.dim_s_obs, params=params)
      # dynamics feature from buffer
      dyn_buff = dynimg(rgb_frames)
      endpoints['dynbuff'] = dyn_buff  # register EP to return dynbuff in forward pass
      dyn_feat, ep_dyn = conv_encoder(dyn_buff, dim_out=params.dim_s_dyn, params=params, scope='DynBuffEncoder')
      endpoints.update(ep_dyn)
      # target feature
      frame_cur = tf.expand_dims(rgb_frame, axis=1)  # introduce sequence axis to concat along
      frame_tgt = tf.expand_dims(tgt_frame, axis=1)  # introduce sequence axis to concat along
      seq_diff = tf.concat([frame_cur, frame_tgt], axis=1)
      dyn_diff = dynimg(seq_diff)
      endpoints['dyndiff'] = dyn_diff  # register EP to return dyndiff in forward pass
      tgt_feat, ep_tgt = conv_encoder(dyn_diff, dim_out=params.dim_s_diff, params=params, scope='DynDiffEncoder')
      endpoints.update(ep_tgt)
      # build state
      state, ep_state = representation_concatenation_v2(
          feat, dyn_feat, jnt_state, tgt_feat, params=params)
      feat_list.append(state)
    else:
      err_msg = "Unknown processing mode for frame buffer: %s!" % (params.proc_obs, )
      raise ValueError(err_msg)
    endpoints.update(ep_enc)
    endpoints.update(ep_state)
    # --- decode feature list with LSTM
    net, ep_dec = lstm_decoder(feat_list, reset, params=params)
    endpoints.update(ep_dec)
    return net, endpoints

# endregion

# region: objective functions

_PREDICTION_KEYS = [
    'cmd_vel',  # [N, 7] (velocity commands for joints)
    'cmd_ee',  # [N, 3] (x-y-z command for EE)
    'cmd_grp',  # [N, 2] (position command for fingers)
    'pos_ee',  # [N, 3] (position prediction for EE)
    'pos_obj',  # [N, 3] (position prediction for object)
]

def mse_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='MSELoss'):
  """
  Computes MSE between the cartesian position and command predictions and
  the respective targets.

  :param predictions: dict mapping _PREDICTION_KEYS to tf.Tensors with predictions
  :param targets: dict mapping _PREDICTION_KEYS to tf.Tensors with target numbers

  :return loss: sum of all losses
  :return endpoints: dict of relevant endpoint nodes
  """
  key_check = set(_PREDICTION_KEYS).issubset(set(predictions.keys())) and \
    set(_PREDICTION_KEYS).issubset(set(targets.keys()))
  assert key_check
  with tf.variable_scope(scope):
    loss_fn = lambda key: tf.losses.mean_squared_error(predictions[key], targets[key])  # TODO: add name to loss term!
    endpoints = dict([('loss_%s' % (key, ), loss_fn(key)) for key in _PREDICTION_KEYS])
    losses = list(endpoints.values())
    loss = tf.add_n(inputs=losses, name='multi_mse_loss')  # TODO: lambdas for loss components
    return loss, endpoints

def cartesian_cmd_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='CartesianCmdLoss'):
  """MSE for cartesian EE command (cmd_ee)."""
  with tf.variable_scope(scope):
    loss = tf.losses.mean_squared_error(predictions['cmd_ee'], targets['cmd_ee'])
    endpoints = {'loss_cmd_ee' : loss}
    return loss, endpoints

def velocity_cmd_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='VelocityCmdLoss'):
  """MSE for joint velocity command (cmd_vel)."""
  with tf.variable_scope(scope):
    loss = tf.losses.mean_squared_error(predictions['cmd_vel'], targets['cmd_vel'])
    endpoints = {'loss_cmd_vel' : loss}
    return loss, endpoints

def gripper_cmd_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='GripperCmdLoss'):
  """Softmax cross-entropy for discrete gripper command (cmd_grp). Assumes cmd_grp to be in {0, 1, 2} tf.int64"""
  with tf.variable_scope(scope):
    # transform to cmd_grp OH (logits_grp: [N, 3]; cmd_grp: [N])
    oh_labels = tf.one_hot(indices=targets['cmd_grp'], depth=params.num_grp_states)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=oh_labels, logits=predictions['logits_cmd_grp'])
    endpoints = {'loss_cmd_grp' : loss}
    return loss, endpoints

def finger_cmd_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='FingerCmdLoss'):
  """MSE for finger position (cmd_grp). Only available for continuous finger commands!"""
  with tf.variable_scope(scope):
    loss = tf.losses.mean_squared_error(predictions['cmd_grp'], targets['cmd_grp'])
    endpoints = {'loss_cmd_grp' : loss}
    return loss, endpoints

def ee_pose_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='EEPoseAuxLoss'):
  """MSE for cartesian EE pose (pos_ee)."""
  with tf.variable_scope(scope):
    loss = tf.losses.mean_squared_error(predictions['pos_ee'], targets['pos_ee'])
    endpoints = {'loss_pos_ee' : loss}
    return loss, endpoints

def obj_pose_loss(
    predictions, targets, params=E2E_VMC_DEFAULT_CONFIG, scope='ObjPoseAuxLoss'):
  """MSE for cartesian object pose (pos_obj)."""
  with tf.variable_scope(scope):
    loss = tf.losses.mean_squared_error(predictions['pos_obj'], targets['pos_obj'])
    endpoints = {'loss_pos_obj' : loss}
    return loss, endpoints

# endregion
