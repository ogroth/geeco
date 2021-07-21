"""TF-Estimator-compatible model functions for E2E-VMC. """

import pprint

import tensorflow as tf

from .graph import e2e_vmc, goal_e2evmc, mse_loss, cartesian_cmd_loss, gripper_cmd_loss, \
  ee_pose_loss, obj_pose_loss
from .utils import count_parameters


#region: public model_fn

def e2evmc_model_fn(features, labels, mode, params):
  """
  Instantiates the model graph for E2E VMC and wraps it in a tf.estimator.EstimatorSpec.

  :param features:
  :param labels:
  :param mode: tf.estimator.ModeKeys (set by estimator APIs)
  :param params:
    'e2evmc_config' : E2EVMCConfig
    'log_steps' : write logs ever n steps
    'debug' : enables additional debug output

  :return spec: tf.estimator.EstimatorSpec
  """
  # decode inputs
  config = params['e2evmc_config']
  if config.img_channels == 3:  # use RGB data
    rgb_frames = tf.identity(features['rgb'], name='in_rgb_frames')
    obs_frames = rgb_frames
  elif config.img_channels == 4:  # use RGB-D data
    rgb_frames = tf.identity(features['rgb'], name='in_rgb_frames')
    depth_frames = tf.identity(features['depth'], name='in_depth_frames')
    obs_frames = tf.concat([rgb_frames, depth_frames], axis=-1, name='in_rgbd_frames')
  else:
    err_msg = "Unsupportd number of channels for input frame: %d!" % config.img_channels
    raise ValueError(err_msg)
  jnt_states = tf.identity(features['jnt_state'], name='in_jnt_states')
  reset = tf.equal(
      tf.reduce_prod(features['step'], name='step_prod'), tf.constant(0, dtype=tf.int64), name='step2reset')
  # graph definition
  endpoints = {}
  net, ep_model = e2e_vmc(obs_frames, jnt_states, reset, params=config)
  endpoints.update(ep_model)
  if config.control_mode == 'cartesian':
    predictions = {
        'cmd_ee' : endpoints['pred_cmd_ee'],
        'logits_cmd_grp' : endpoints['logits_cmd_grp'],
        'pos_ee' : endpoints['pred_aux_ee'],
        'pos_obj' : endpoints['pred_aux_obj'],
    }
  elif config.control_mode == 'velocity':
    predictions = {
        'cmd_vel' : endpoints['pred_cmd_vel'],
        'cmd_ee' : endpoints['pred_cmd_ee'],
        'cmd_grp' : endpoints['pred_cmd_grp'],
        'pos_ee' : endpoints['pred_aux_ee'],
        'pos_obj' : endpoints['pred_aux_obj'],
    }
  # loss function
  if mode != tf.estimator.ModeKeys.PREDICT:
    # weight regularization
    with tf.variable_scope('RegularizationLoss'):
      loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_reg_loss')
      tf.add_to_collection(tf.GraphKeys.LOSSES, loss_reg)
      endpoints['loss_reg'] = loss_reg
    if config.control_mode == 'cartesian':
      targets = {
          'cmd_ee' : labels['cmd'][:, :3],  # delta in x-y-z
          'cmd_grp' : labels['cmd'][:, 3],  # gripper action
          'pos_ee' : features['ee_state'][:, -1, :3],  # from last frame of window, only x-y-z
          'pos_obj' : features['obj_state'][:, -1, :3],  # from last frame of window, only x-y-z
      }
      # transform cmd_grp: to int -> {-1, 0, 1} -> {0, 1, 2}
      cmd_grp = targets['cmd_grp']
      cmd_grp = tf.dtypes.cast(tf.math.rint(cmd_grp), tf.int32)  # cast to int
      cmd_grp += 1  # shift class indices by one
      targets['cmd_grp'] = cmd_grp
      # define losses
      loss_cmd_ee, ep_loss_cmd_ee = cartesian_cmd_loss(predictions, targets, config)
      loss_cmd_grp, ep_loss_cmd_grp = gripper_cmd_loss(predictions, targets, config)
      loss_pos_ee, ep_loss_pos_ee = ee_pose_loss(predictions, targets, config)
      loss_pos_obj, ep_loss_pos_obj = obj_pose_loss(predictions, targets, config)
      loss = tf.add_n(inputs=[loss_cmd_ee, loss_cmd_grp], name='command_loss') \
          + config.lambda_aux * tf.add_n(inputs=[loss_pos_ee, loss_pos_obj], name='pose_loss')
      ep_losses = {}
      for eps in [ep_loss_cmd_ee, ep_loss_cmd_grp, ep_loss_pos_ee, ep_loss_pos_obj]:
        ep_losses.update(eps)
    elif config.control_mode == 'velocity':
      targets = {
          'cmd_vel' : labels['vel_target'],
          'cmd_ee' : labels['ee_target'][:, :3],  # only x-y-z
          'cmd_grp' : labels['grp_target'],
          'pos_ee' : features['ee_state'][:, -1, :3],  # from last frame of window, only x-y-z
          'pos_obj' : features['obj_state'][:, -1, :3],  # from last frame of window, only x-y-z
      }
      loss, ep_losses = mse_loss(predictions, targets)
    # final loss: control-space dependent loss + regularization
    loss = tf.add(loss, loss_reg)
    endpoints.update(ep_losses)
  # optimizer
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  # eval metrics
  if mode == tf.estimator.ModeKeys.EVAL:
    if config.control_mode == 'cartesian':
      eval_metric_op = lambda pred, target: tf.metrics.mean_squared_error(labels=target, predictions=pred)
      eval_metric_keys = ['cmd_ee', 'pos_ee', 'pos_obj']
      eval_metric_ops = dict([(k, eval_metric_op(predictions[k], targets[k])) for k in eval_metric_keys])
      eval_metric_ops['cmd_grp'] = tf.metrics.accuracy(
          labels=targets['cmd_grp'],
          predictions=tf.argmax(predictions['logits_cmd_grp'], axis=-1)
      )
    elif config.control_mode == 'velocity':
      eval_metric_op = lambda pred, target: tf.metrics.mean_squared_error(labels=target, predictions=pred)
      eval_metric_keys = ['cmd_vel', 'cmd_ee', 'cmd_grp', 'pos_ee', 'pos_obj']
      eval_metric_ops = dict([(k, eval_metric_op(predictions[k], targets[k])) for k in eval_metric_keys])
  # print graph summary, TODO: debug logger!
  print(">>> Graph Summary (%d trainable parameters):" % (count_parameters(), ))
  pprint.pprint(endpoints)
  # register run hooks
  run_hooks = []
  summary_saver_hooks = _create_loss_part_summary_hooks(logging_interval=params['log_steps'])
  run_hooks.extend(summary_saver_hooks)
  # create estimator spec
  if mode == tf.estimator.ModeKeys.TRAIN:
    spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, training_hooks=run_hooks)
  elif mode == tf.estimator.ModeKeys.EVAL:
    spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=run_hooks)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions)
  else:
    err_msg = "Unknown estimator mode: %s" % (mode, )
    raise RuntimeError(err_msg)
  return spec

# TODO: refactor and merge with e2evmc (goal condition as parameter)
def goal_e2evmc_model_fn(features, labels, mode, params):
  """
  Instantiates the model graph for goal-conditioned E2EVMC and wraps it in a
  tf.estimator.EstimatorSpec.

  :param features:
  :param labels:
  :param mode: tf.estimator.ModeKeys (set by estimator APIs)
  :param params:
    'e2evmc_config' : E2EVMCConfig
    'log_steps' : write logs ever n steps
    'debug' : enables additional debug output

  :return spec: tf.estimator.EstimatorSpec
  """
  # decode inputs
  config = params['e2evmc_config']
  if config.img_channels == 3:  # use RGB data
    rgb_frames = tf.identity(features['rgb'], name='in_rgb_frames')
    obs_frames = rgb_frames
    tgt_rgb_frame = tf.identity(features['target_rgb'], name='tgt_rgb_frame')
    tgt_frame = tgt_rgb_frame
  elif config.img_channels == 4:  # use RGB-D data
    rgb_frames = tf.identity(features['rgb'], name='in_rgb_frames')
    depth_frames = tf.identity(features['depth'], name='in_depth_frames')
    obs_frames = tf.concat([rgb_frames, depth_frames], axis=-1, name='in_rgbd_frames')
    tgt_rgb_frame = tf.identity(features['target_rgb'], name='tgt_rgb_frame')
    tgt_depth_frame = tf.identity(features['target_depth'], name='tgt_depth_frame')
    tgt_frame = tf.concat([tgt_rgb_frame, tgt_depth_frame], axis=-1, name='tgt_rgbd_frame')
  else:
    err_msg = "Unsupported number of channels for input frame: %d!" % config.img_channels
    raise ValueError(err_msg)
  jnt_states = tf.identity(features['jnt_state'], name='in_jnt_states')
  reset = tf.equal(
      tf.reduce_prod(features['step'], name='step_prod'), tf.constant(0, dtype=tf.int64), name='step2reset')
  # graph definition
  endpoints = {}
  net, ep_model = goal_e2evmc(obs_frames, jnt_states, tgt_frame, reset, params=config)
  endpoints.update(ep_model)
  if config.control_mode == 'cartesian':
    predictions = {
        'cmd_ee' : endpoints['pred_cmd_ee'],
        'logits_cmd_grp' : endpoints['logits_cmd_grp'],
        'pos_ee' : endpoints['pred_aux_ee'],
        'pos_obj' : endpoints['pred_aux_obj'],
    }
  elif config.control_mode == 'velocity':
    predictions = {
        'cmd_vel' : endpoints['pred_cmd_vel'],
        'cmd_ee' : endpoints['pred_cmd_ee'],
        'cmd_grp' : endpoints['pred_cmd_grp'],
        'pos_ee' : endpoints['pred_aux_ee'],
        'pos_obj' : endpoints['pred_aux_obj'],
    }
  # loss function
  if mode != tf.estimator.ModeKeys.PREDICT:
    # weight regularization
    with tf.variable_scope('RegularizationLoss'):
      loss_reg = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_reg_loss')
      tf.add_to_collection(tf.GraphKeys.LOSSES, loss_reg)
      endpoints['loss_reg'] = loss_reg
    if config.control_mode == 'cartesian':
      targets = {
          'cmd_ee' : labels['cmd'][:, :3],  # delta in x-y-z
          'cmd_grp' : labels['cmd'][:, 3],  # gripper action
          'pos_ee' : features['ee_state'][:, -1, :3],  # from last frame of window, only x-y-z
          'pos_obj' : features['obj_state'][:, -1, :3],  # from last frame of window, only x-y-z
      }
      # transform cmd_grp: to int -> {-1, 0, 1} -> {0, 1, 2}
      cmd_grp = targets['cmd_grp']
      cmd_grp = tf.dtypes.cast(tf.math.rint(cmd_grp), tf.int32)  # cast to int
      cmd_grp += 1  # shift class indices by one
      targets['cmd_grp'] = cmd_grp
      # define losses
      loss_cmd_ee, ep_loss_cmd_ee = cartesian_cmd_loss(predictions, targets, config)
      loss_cmd_grp, ep_loss_cmd_grp = gripper_cmd_loss(predictions, targets, config)
      loss_pos_ee, ep_loss_pos_ee = ee_pose_loss(predictions, targets, config)
      loss_pos_obj, ep_loss_pos_obj = obj_pose_loss(predictions, targets, config)
      # loss = tf.add_n(
      #     inputs=[loss_cmd_ee, loss_cmd_grp, loss_pos_ee, loss_pos_obj], name='hybrid_loss')
      loss = tf.add_n(inputs=[loss_cmd_ee, loss_cmd_grp], name='command_loss') \
          + config.lambda_aux * tf.add_n(inputs=[loss_pos_ee, loss_pos_obj], name='pose_loss')
      ep_losses = {}
      for eps in [ep_loss_cmd_ee, ep_loss_cmd_grp, ep_loss_pos_ee, ep_loss_pos_obj]:
        ep_losses.update(eps)
    elif config.control_mode == 'velocity':
      targets = {
          'cmd_vel' : labels['vel_target'],
          'cmd_ee' : labels['ee_target'][:, :3],  # only x-y-z
          'cmd_grp' : labels['grp_target'],
          'pos_ee' : features['ee_state'][:, -1, :3],  # from last frame of window, only x-y-z
          'pos_obj' : features['obj_state'][:, -1, :3],  # from last frame of window, only x-y-z
      }
      loss, ep_losses = mse_loss(predictions, targets)
    # final loss: control-space dependent loss + regularization
    loss = tf.add(loss, loss_reg)
    endpoints.update(ep_losses)
  # optimizer
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.AdamOptimizer(learning_rate=config.lr)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
  # eval metrics
  if mode == tf.estimator.ModeKeys.EVAL:
    if config.control_mode == 'cartesian':
      eval_metric_op = lambda pred, target: tf.metrics.mean_squared_error(labels=target, predictions=pred)
      eval_metric_keys = ['cmd_ee', 'pos_ee', 'pos_obj']
      eval_metric_ops = dict([(k, eval_metric_op(predictions[k], targets[k])) for k in eval_metric_keys])
      eval_metric_ops['cmd_grp'] = tf.metrics.accuracy(
          labels=targets['cmd_grp'],
          predictions=tf.argmax(predictions['logits_cmd_grp'], axis=-1)
      )
    elif config.control_mode == 'velocity':
      eval_metric_op = lambda pred, target: tf.metrics.mean_squared_error(labels=target, predictions=pred)
      eval_metric_keys = ['cmd_vel', 'cmd_ee', 'cmd_grp', 'pos_ee', 'pos_obj']
      eval_metric_ops = dict([(k, eval_metric_op(predictions[k], targets[k])) for k in eval_metric_keys])
  # print graph summary, TODO: debug logger!
  print(">>> Graph Summary (%d trainable parameters):" % (count_parameters(), ))
  pprint.pprint(endpoints)
  # register run hooks
  run_hooks = []
  summary_saver_hooks = _create_loss_part_summary_hooks(logging_interval=params['log_steps'])
  run_hooks.extend(summary_saver_hooks)
  # create estimator spec
  if mode == tf.estimator.ModeKeys.TRAIN:
    spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op, training_hooks=run_hooks)
  elif mode == tf.estimator.ModeKeys.EVAL:
    spec = tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=run_hooks)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions)
  else:
    err_msg = "Unknown estimator mode: %s" % (mode, )
    raise RuntimeError(err_msg)
  return spec

#endregion

#region: private logging and summary functions
# loss, eval metric and training speed are logged by default by the estimator
# logging hooks can only be created within e2evmc_model_fn() !

_DEFAULT_LOGGING_INTERVAL = 1000  # num. of iterations before updating logging hooks

# TODO: only applies to velocity control
def _create_prediction_logging_hook(logging_interval=_DEFAULT_LOGGING_INTERVAL, model_scope='VMC'):
  """Creates a logging hook for the model predictions."""
  log_map = {  # log tag -> tensor name
      'pred_cmd_vel' : '%s/LSTMDecoder/pred_cmd_vel/BiasAdd:0' % model_scope,
      'pred_cmd_ee' : '%s/LSTMDecoder/pred_cmd_ee/BiasAdd:0' % model_scope,
      'pred_cmd_grp' : '%s/LSTMDecoder/pred_cmd_grp/BiasAdd:0' % model_scope,
      'pred_aux_ee' : '%s/LSTMDecoder/pred_aux_ee/BiasAdd:0' % model_scope,
      'pred_aux_obj' : '%s/LSTMDecoder/pred_aux_obj/BiasAdd:0' % model_scope,
  }
  logging_hook = tf.train.LoggingTensorHook(
      tensors=log_map, every_n_iter=logging_interval)
  return logging_hook

# TODO: define logging hook for cartesian control

def _create_loss_part_summary_hooks(logging_interval=_DEFAULT_LOGGING_INTERVAL):
  """Creates logging hooks for all parts of the loss function."""
  loss_ops = tf.get_collection(tf.GraphKeys.LOSSES)
  loss_summary_hooks = []
  for loss in loss_ops:
    loss_summary = tf.summary.scalar(name=loss.name, tensor=loss)
    summary_saver_hook = tf.train.SummarySaverHook(summary_op=loss_summary, save_steps=logging_interval)
    loss_summary_hooks.append(summary_saver_hook)
  return loss_summary_hooks

def _create_input_summary_hook(logging_interval=_DEFAULT_LOGGING_INTERVAL):
  """Creates a logging hook for the input data."""
  rgb_frames = tf.get_default_graph().get_tensor_by_name('in_rgb_frames')  # TODO: incorrect retrieval!
  summary_op = tf.summary.image(tensor=rgb_frames, max_outputs=1, name='in_rgb_frames')
  summary_hook = tf.train.SummarySaverHook(summary_op=summary_op, save_steps=logging_interval)
  return summary_hook

#endregion
