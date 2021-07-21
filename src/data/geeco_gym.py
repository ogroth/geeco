"""
Tfrecord-based recorder and provider for stream data recorded out of GEECO gyms.
"""

import os
import re
import json
from collections import namedtuple

import numpy as np
from PIL import Image
import scipy.misc
import tensorflow as tf

from .data_recorder import TfrSequenceEncoding
from .utils.tfrecord import convert_to_feature


# ---------- constants ----------

# @deprecated: V1 encoding
_IMG_H, _IMG_W, _IMG_C = 256, 256, 3  # image dimensions
_NUM_ACT = 2  # number of actuated joints (2-point gripper)

_MODE_KEYS = {
    'train' : tf.estimator.ModeKeys.TRAIN,
    'val' : tf.estimator.ModeKeys.EVAL,
    'test' : tf.estimator.ModeKeys.PREDICT,
}


# ---------- data structures ----------

PickAndPlaceMetaV4 = namedtuple(
    typename='PickAndPlaceMetaV4',
    field_names=[
        # general episode information
        'episode_length',  # int
        # observation data
        'img_height',  # int
        'img_width',  # int
        # proprioception data
        'monitored_joints',  # [str]
        'actuated_joints',  # [str]
        'monitored_mocaps',  # [str]
        'monitored_objects',  # [str]
        'dim_cmd',  # length of command vector (as specified by env), int
        'dim_ctrl',  # length of actuator control vector, int
    ])


# ---------- data encoding ----------

class PickAndPlaceEncodingV4(TfrSequenceEncoding):
  """
  Encodes and decodes the data sequence of a PickAndPlace experiment executed by
  a Fetch robot in GEECO Gym.

  data = {
    'episode_length' : scalar(int),
    'img_height' : scalar(int),
    'img_width' : scalar(int),
    'monitored_joints' : np.ndarray(string),  # names of recorded joints
    'actuated_joints' : np.ndarray(string),  # names of joints actuated by motors
    'monitored_mocaps' : np.ndarray(string),  # names of recorded mocaps
    'monitored_objects' : np.ndarray(string),  # names of recorded objects (freejoint-based)
    'dim_cmd' : scalar(int),  # length of the command vector (env)
    'dim_ctrl' : scalar(int),  # length of the control vector (sim)
    'task_goal' : scalar(str),  # name of goal object
    'task_object' : scalar(str),  # name of manipulated object
    'sequence' : [
      { # one data frame
        'step' : int,  # step (frame ID)
        'ts' : float,  # timestamp [s]
        'rgb' : np.ndarray(float),  # [img_height, img_width, 3]
        'depth : np.ndarray(float),  # [img_height, img_width, 1]
        'cmd' : np.ndarray(float),  # command array
        'ctrl' : np.ndarray(float),  # control array
        'joint_qpos_<jnt_name>' : np.ndarray(float),  # qpos [7] of tracked <jnt_name>
        'joint_qvel_<jnt_name>' : np.ndarray(float),  # qvel [7] of tracked <jnt_name>
        'mocap_qpos_<mcp_name>' : np.ndarray(float),  # qpos [7] of tracked <mcp_name>
        'object_qpos_<obj_jnt_name>' : np.ndarray(float),  # qpos [7] of tracked <obj_jnt_name>
        'goal_pos' : np.ndarray(float),  # qpos [7] of tracked goal object
        'obj_pos' : np.ndarray(float),  # qpos [7] of tracked manipulated object
      },
      {
        ...
      }
    ]
  }
  """

  def __init__(self, meta: PickAndPlaceMetaV4):
    """
    Initializes a PickAndPlaceEncoding by defining the data structure keys
    based on the names of the monitored joints.

    :param meta: meta struct
    """
    super().__init__()
    # --- keys setup
    context_keys = list(meta._asdict().keys())
    context_keys += ['task_goal', 'task_object']
    self._context_keys = context_keys
    frame_keys = ['step', 'ts', 'rgb', 'depth', 'cmd', 'ctrl','goal_qpos', 'obj_qpos']
    for jnt_name in meta.monitored_joints:
      frame_keys.append('joint_qpos-%s' % (jnt_name, ))
      frame_keys.append('joint_qvel-%s' % (jnt_name, ))
    for mcp_name in meta.monitored_mocaps:
      frame_keys.append('mocap_qpos-%s' % (mcp_name, ))
    for obj_jnt_name in meta.monitored_objects:
      frame_keys.append('object_qpos-%s' % (obj_jnt_name, ))
    self._frame_keys = frame_keys
    # --- custom class members
    self._meta = meta  # struct containing the encoding's meta info

  def decode(self):
    """
    Provides context and sequence decoders to be passed into
    tf.parse_single_sequence_example(serialized_example, context_decoder, sequence_decoder).
    """
    context_decoder = {
        'episode_length' : tf.FixedLenFeature([], dtype=tf.int64),
        'img_height' : tf.FixedLenFeature([], dtype=tf.int64),
        'img_width' : tf.FixedLenFeature([], dtype=tf.int64),
        'monitored_joints' : tf.FixedLenFeature([len(self._meta.monitored_joints)], dtype=tf.string),
        'actuated_joints' : tf.FixedLenFeature([len(self._meta.actuated_joints)], dtype=tf.string),
        'monitored_mocaps' : tf.FixedLenFeature([len(self._meta.monitored_mocaps)], dtype=tf.string),
        'monitored_objects' : tf.FixedLenFeature([len(self._meta.monitored_objects)], dtype=tf.string),
        'dim_cmd' : tf.FixedLenFeature([], dtype=tf.int64),
        'dim_ctrl' : tf.FixedLenFeature([], dtype=tf.int64),
        'task_goal' : tf.FixedLenFeature([], dtype=tf.string),
        'task_object' : tf.FixedLenFeature([], dtype=tf.string),
    }
    sequence_decoder = {
        'step' : tf.FixedLenSequenceFeature([], dtype=tf.int64),
        'ts' : tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'rgb' : tf.FixedLenSequenceFeature([self._meta.img_height * self._meta.img_width * 3], dtype=tf.float32),
        'depth' : tf.FixedLenSequenceFeature([self._meta.img_height * self._meta.img_width * 1], dtype=tf.float32),
        'cmd' : tf.FixedLenSequenceFeature([self._meta.dim_cmd], dtype=tf.float32),
        'ctrl' : tf.FixedLenSequenceFeature([self._meta.dim_ctrl], dtype=tf.float32),
        'obj_qpos' : tf.FixedLenSequenceFeature([7], dtype=tf.float32),
        'goal_qpos' : tf.FixedLenSequenceFeature([7], dtype=tf.float32),
    }
    jnt_qpos_keys = [k for k in self._frame_keys if k.startswith('joint_qpos-')]
    jnt_qvel_keys = [k for k in self._frame_keys if k.startswith('joint_qvel-')]
    mcp_qpos_keys = [k for k in self._frame_keys if k.startswith('mocap_qpos-')]
    obj_qpos_keys = [k for k in self._frame_keys if k.startswith('object_qpos-')]
    pos_keys = [*jnt_qpos_keys]
    qvel_keys = [*jnt_qvel_keys]
    qpos_keys = [*mcp_qpos_keys, *obj_qpos_keys]
    for k in pos_keys:
      sequence_decoder[k] = tf.FixedLenSequenceFeature([], dtype=tf.float32)
    for k in qvel_keys:
      sequence_decoder[k] = tf.FixedLenSequenceFeature([], dtype=tf.float32)
    for k in qpos_keys:
      sequence_decoder[k] = tf.FixedLenSequenceFeature([7], dtype=tf.float32)
    return context_decoder, sequence_decoder


# ---------- helper functions ----------

# TODO: merge target frame loading functions

def load_target_frames(dataset_dir, tfrecord_name, load_depth=True):
  """
  Loads keyframes, if available.
  Otherwise tries to load single target frame for episode.
  If neither are available, returns empty list.
  """
  record_id = re.search(r'\d+', tfrecord_name).group(0)
  keyframe_filename = 'key_frames_%s.json' % (record_id, )
  keyframe_file = os.path.join(dataset_dir, 'data', keyframe_filename)
  if os.path.exists(keyframe_file):
    return load_keyframes(dataset_dir, tfrecord_name)
  else:
    return [load_target_frame(dataset_dir, tfrecord_name, load_depth)]

def load_target_frame(dataset_dir, tfrecord_name, load_depth=True):
  """Loads the target frame as RGB-D tensor for the given tfrecord_name."""
  filename = os.path.basename(tfrecord_name).split('.')[0]
  rgb_target_path = os.path.join(dataset_dir, 'images', 'targets', 'rgb', filename + '.png')
  # rgb_import = scipy.misc.imread(rgb_target_path) / 255.0
  rgb_import = np.array(Image.open(rgb_target_path), dtype=np.float32) / 255.0
  if load_depth:
    depth_target_path = os.path.join(dataset_dir, 'images', 'targets', 'depth', filename + '.npy')
    depth_import = np.expand_dims(np.load(depth_target_path), axis=-1)
    target_frame = np.concatenate([rgb_import, depth_import], axis=-1)
  else:
    target_frame = rgb_import
  print("Read %s" % filename)  # DEBUG
  return target_frame

def load_keyframes(dataset_dir, tfrecord_name):
  """Loads the target frame as RGB-D tensor for the given tfrecord_name."""
  filename = os.path.basename(tfrecord_name).split('.')[0]
  rgb_keyframe_dir = os.path.join(dataset_dir, 'images', 'keyframes', 'rgb')
  depth_keyframe_dir = os.path.join(dataset_dir, 'images', 'keyframes', 'depth')
  rgb_keyframe_files = sorted([f for f in os.listdir(rgb_keyframe_dir) if f.startswith(filename)])
  depth_keyframe_files = sorted([f for f in os.listdir(depth_keyframe_dir) if f.startswith(filename)])
  keyframes = []
  for rgb_file, depth_file in zip(rgb_keyframe_files, depth_keyframe_files):
    rgb_path = os.path.join(dataset_dir, 'images', 'keyframes', 'rgb', rgb_file)
    depth_path = os.path.join(dataset_dir, 'images', 'keyframes', 'depth', depth_file)
    # rgb_import = scipy.misc.imread(rgb_path) / 255.0
    rgb_import = np.array(Image.open(rgb_path), dtype=np.float32) / 255.0
    depth_import = np.expand_dims(np.load(depth_path), axis=-1)
    target_frame = np.concatenate([rgb_import, depth_import], axis=-1)
    keyframes.append(target_frame)
    # print("Read %s" % filename)  # DEBUG
  return keyframes

def load_intermediate_target_frames(dataset_dir, tfrecord_name):
  """Loads the intermediate target frames as RGB-D tensor for the given tfrecord_name."""
  filename = os.path.basename(tfrecord_name).split('.')[0]
  rgb_target_dir = os.path.join(dataset_dir, 'images', 'inter_targets', 'rgb')
  depth_target_dir = os.path.join(dataset_dir, 'images', 'inter_targets', 'depth')
  rgb_target_paths = [os.path.join(rgb_target_dir, f) for f in os.listdir(rgb_target_dir) if f.startswith(filename)]
  rgb_target_paths.sort()
  depth_target_paths = [os.path.join(depth_target_dir, f) for f in os.listdir(depth_target_dir) if f.startswith(filename)]
  depth_target_paths.sort()
  target_frames = []
  for rgb_target_path, depth_target_path in zip(rgb_target_paths, depth_target_paths):
    # rgb_import = scipy.misc.imread(rgb_target_path) / 255.0
    rgb_import = np.array(Image.open(rgb_target_path), dtype=np.float32) / 255.0
    depth_import = np.expand_dims(np.load(depth_target_path), axis=-1)
    target_frame = np.concatenate([rgb_import, depth_import], axis=-1)
    target_frames.append(target_frame)
  return target_frames


# ---------- global input_fn ----------

def pickplace_input_fn(
    dataset_dir,
    split_name,
    mode,  # train | val | test
    encoding='v4',  # version of the data encoding; available: v1 | v2 | v3 | v4
    window_size=4,  # length of fetched (sub-)sequences, max. episode length (=100)
    fetch_target=False,  # also fetch target images, 20% performance drop!
    shuffle_buffer=128,  # number of elements to shuffle next batch from
    batch_size=1,
    num_epochs=1,
    num_threads=4,  # number of parallel threads for data preparation
    prefetch_size=4,  # number of batches to prefetch into the pipeline
    seed=None):
  """
  Input_fn for a dataset of tfrecords with `PickAndPlaceEncodingV*`.
  Assumed dataset structure:
  <dataset_dir>/
    meta/
      meta_info.json
      init-*.csv
    data/
      *.tfrecord.zlib
      *.pkl
    splits/
      <split_name>/
        train.txt
        eval.txt
        test.txt
        init-train.csv
        init-eval.csv
        init-test.csv
  """
  input_fn = _ENCODING2INPUTFN[encoding]
  return input_fn(
      dataset_dir,
      split_name,
      mode,
      window_size,
      fetch_target,  # TODO: implement for v1 & v2!
      shuffle_buffer,
      batch_size,
      num_epochs,
      num_threads,
      prefetch_size,
      seed)


# ---------- input_fn V4 ----------

def _get_meta_v4(data_dir):
  """Reads the meta information of the dataset."""
  meta_info_path = os.path.join(data_dir, 'meta', 'meta_info.json')
  with open(meta_info_path, 'r') as fp:
    meta_info_dict = json.load(fp)
  meta = PickAndPlaceMetaV4(**meta_info_dict)
  return meta

def _parse_v4(proto_example, fetch_target, meta):
  """Decode frame sequence from tfrecord and normalize values."""
  # decoder setup
  encoding = PickAndPlaceEncodingV4(meta)
  context_decoder, sequence_decoder = encoding.decode()
  # parse proto example
  context_data, sequence_data = tf.parse_single_sequence_example(
      serialized=proto_example,
      context_features=context_decoder,
      sequence_features=sequence_decoder)
  parsed_example = {}
  # parsed_example.update(context_data)  # context == meta here!
  parsed_example.update(sequence_data)
  # reshape data fields
  rgb = parsed_example['rgb']
  parsed_example['rgb'] = tf.reshape(rgb, [-1, meta.img_height, meta.img_width, 3])
  depth = parsed_example['depth']
  parsed_example['depth'] = tf.reshape(depth, [-1, meta.img_height, meta.img_width, 1])
  # normalize data
  parsed_example['rgb'] /= 255.0  # RGB recorded as uint8 [0 .. 255]
  # optionally fetch target
  if fetch_target:
    parsed_example['target_rgb'] = tf.expand_dims(parsed_example['rgb'][-1], axis=0)
    parsed_example['target_depth'] = tf.expand_dims(parsed_example['depth'][-1], axis=0)
  return parsed_example

def _preprocess_states_v4(example, fetch_target):
  """Concatenate the frame annotations into state vectors."""
  processed_example = {
      'step' : example['step'],
      'ts' : example['ts'],
      'rgb' : example['rgb'],
      'depth' : example['depth'],
      'cmd' : example['cmd'],
      'ctrl' : example['ctrl'],
      'jnt_state' : None,
      'vel_state' : None,
      'ee_state' : example['mocap_qpos-robot0:mocap'],
      'grp_state' : None,
      'goal_state' : example['goal_qpos'],
      'obj_state' : example['obj_qpos'],
  }
  if fetch_target:
    processed_example['target_rgb'] = example['target_rgb']
    processed_example['target_depth'] = example['target_depth']

  jnt_state = tf.stack(
      values=[
          example['joint_qpos-robot0:shoulder_pan_joint'],
          example['joint_qpos-robot0:shoulder_lift_joint'],
          example['joint_qpos-robot0:upperarm_roll_joint'],
          example['joint_qpos-robot0:elbow_flex_joint'],
          example['joint_qpos-robot0:forearm_roll_joint'],
          example['joint_qpos-robot0:wrist_flex_joint'],
          example['joint_qpos-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_jnt_state')
  processed_example['jnt_state'] = jnt_state
  vel_state = tf.stack(
      values=[
          example['joint_qvel-robot0:shoulder_pan_joint'],
          example['joint_qvel-robot0:shoulder_lift_joint'],
          example['joint_qvel-robot0:upperarm_roll_joint'],
          example['joint_qvel-robot0:elbow_flex_joint'],
          example['joint_qvel-robot0:forearm_roll_joint'],
          example['joint_qvel-robot0:wrist_flex_joint'],
          example['joint_qvel-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_vel_state')
  processed_example['vel_state'] = vel_state
  grp_state = tf.stack(
      values=[
          example['joint_qpos-robot0:l_gripper_finger_joint'],
          example['joint_qpos-robot0:r_gripper_finger_joint'],
      ],
      axis=1,
      name='stack_grp_state')
  processed_example['grp_state'] = grp_state
  return processed_example

def _prepare_v4(window_example, fetch_target):
  """Transforms each example window into a (feature, label) tuple."""
  feature = {
      'step' : window_example['step'],  # [K]
      'ts' : window_example['ts'],  # [K]
      'rgb' : window_example['rgb'],  # [K, IMG_H, IMG_W, 3]
      'depth' : window_example['depth'],  # [K, IMG_H, IMG_W, 1]
      'jnt_state' : window_example['jnt_state'],  # [K, 7]: pos of all 7 joints of arm
      'vel_state' : window_example['vel_state'],  # [K, 7]: vel of all 7 joints of arm
      'ee_state' : window_example['ee_state'],  # [K, 7]: absolute qpos of EE
      'grp_state' : window_example['grp_state'],  # [K, 2]: absolute pos of each finger
      'goal_state' : window_example['goal_state'],  # [K, 7]: absolute qpos of target object
      'obj_state' : window_example['obj_state'],  # [K, 7]: absolute qpos of moved object
      'cmd' : window_example['cmd'],  # [K, 4]: commands given at each frame
      'ctrl' : window_example['ctrl'],  # [K, 2]: control vector at each frame
  }
  if fetch_target:
    feature['target_rgb'] = window_example['target_rgb']  # [IMG_H, IMG_W, 3]
    feature['target_depth'] = window_example['target_depth']  # [IMG_H, IMG_W, 1]
  label = {
      'cmd' : window_example['cmd'][-1],  # [4]: cmd given to the agent
      'ctrl' : window_example['ctrl'][-1],  # [2]: ctrl for each finger actuator
      'vel_target' : window_example['vel_target'][-1],  # [7]: absolute target velocities for all 7 joints of arm
      'ee_target' : window_example['ee_target'][-1],  # [7]: absolute target qpos for EE
      'grp_target' : window_example['grp_target'][-1],  # [2]: absolute target positions for fingers
  }
  return (feature, label)

def pickplace_input_fn_v4(
    dataset_dir,
    split_name,
    mode,  # train | val | test
    window_size=4,  # length of fetched (sub-)sequences, max. episode length (=100)
    fetch_target=False,  # also fetch target images, 20% performance drop!
    shuffle_buffer=128,  # number of elements to shuffle next batch from
    batch_size=1,
    num_epochs=1,
    num_threads=4,  # number of parallel threads for data preparation
    prefetch_size=4,  # number of batches to prefetch into the pipeline
    seed=None):
  """
  Input_fn for a dataset of tfrecords with `PickAndPlaceEncodingV2`.
  Assumed dataset structure:
  <dataset_dir>/
    meta/
      meta_info.json
      init-*.csv
    data/
      *.tfrecord.zlib
      *.pkl
    splits/
      <split_name>/
        train.txt
        eval.txt
        test.txt
        init-train.csv
        init-eval.csv
        init-test.csv
  """
  # collect tfrecords
  meta = _get_meta_v4(dataset_dir)
  # print(meta)
  tfrecord_paths = _collect_tfrecords_v2(dataset_dir, split_name, mode)
  if mode == 'train':  # shuffle on sequence-level
    np.random.shuffle(tfrecord_paths)
  # print(tfrecord_paths)
  num_tfrecords = len(tfrecord_paths)
  print("[pickplace_input_fn_v4] #tfrecords: %d" % num_tfrecords)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(  # set of proto strings
      filenames=tfrecord_paths,
      compression_type='ZLIB',  # TODO: infer compression type from tfrecords
      num_parallel_reads=num_threads)
  # shuffle data for training
  # if mode == 'train':  # shuffle on sequence-level
  #   dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
  # parse tfrecords
  dataset = dataset.map(
      lambda proto_example: _parse_v4(proto_example, fetch_target, meta),
      num_parallel_calls=num_threads)
  # transform state vectors
  dataset = dataset.map(
      lambda example: _preprocess_states_v4(example, fetch_target),
      num_parallel_calls=num_threads)
  # transform command vectors
  dataset = dataset.map(
      lambda example: _preprocess_targets_v3(example),
      num_parallel_calls=num_threads)
  # create sliding windows
  dataset = dataset.map(
      lambda example: _window_v3(example, meta, window_size=window_size),
      num_parallel_calls=num_threads)
  dataset = dataset.apply(tf.data.experimental.unbatch())  # unbatch frame windows
  # prepare data as supervised tuples: (feature, label)
  dataset = dataset.map(
      lambda window_example: _prepare_v4(window_example, fetch_target),
      num_parallel_calls=num_threads)
  # set up batching
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(prefetch_size)
  return dataset

def pickplace_dataset_v4(dataset_dir, num_threads=4):
  """
  Returns a dataset of tfrecords with `PickAndPlaceEncodingV3`.
  Assumed dataset structure:
  <dataset_dir>/
    meta/
      meta_info.json
      init-*.csv
    data/
      *.tfrecord.zlib
      *.pkl
    splits/
      <split_name>/
        train.txt
        eval.txt
        test.txt
        init-train.csv
        init-eval.csv
        init-test.csv
  """
  # collect tfrecords
  meta = _get_meta_v4(dataset_dir)
  # print(meta)
  tfrecord_paths = _collect_tfrecords_v2(dataset_dir, split_name=None, mode=None)
  # print(tfrecord_paths)
  num_tfrecords = len(tfrecord_paths)
  print("#tfrecords: %d" % num_tfrecords)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(  # set of proto strings
      filenames=tfrecord_paths,
      compression_type='ZLIB',  # TODO: infer compression type from tfrecords
      num_parallel_reads=num_threads)
  # parse tfrecords
  dataset = dataset.map(
      lambda proto_example: _parse_v4(proto_example, fetch_target=False, meta=meta),
      num_parallel_calls=num_threads)
  return dataset


# ---------- input_fn V3 ----------

def _parse_v3(proto_example, fetch_target, meta):
  """Decode frame sequence from tfrecord and normalize values."""
  # decoder setup
  encoding = PickAndPlaceEncodingV3(meta)
  context_decoder, sequence_decoder = encoding.decode()
  # parse proto example
  context_data, sequence_data = tf.parse_single_sequence_example(
      serialized=proto_example,
      context_features=context_decoder,
      sequence_features=sequence_decoder)
  parsed_example = {}
  # parsed_example.update(context_data)  # context == meta here!
  parsed_example.update(sequence_data)
  # reshape data fields
  rgb = parsed_example['rgb']
  parsed_example['rgb'] = tf.reshape(rgb, [-1, meta.img_height, meta.img_width, 3])
  depth = parsed_example['depth']
  parsed_example['depth'] = tf.reshape(depth, [-1, meta.img_height, meta.img_width, 1])
  # normalize data
  parsed_example['rgb'] /= 255.0  # RGB recorded as uint8 [0 .. 255]
  # optionally fetch target
  if fetch_target:
    parsed_example['target_rgb'] = tf.expand_dims(parsed_example['rgb'][-1], axis=0)
    parsed_example['target_depth'] = tf.expand_dims(parsed_example['depth'][-1], axis=0)
  return parsed_example

def _preprocess_states_v3(example, fetch_target):
  """Concatenate the frame annotations into state vectors."""
  processed_example = {
      'step' : example['step'],
      'ts' : example['ts'],
      'rgb' : example['rgb'],
      'depth' : example['depth'],
      'cmd' : example['cmd'],
      'jnt_state' : None,
      'vel_state' : None,
      'ee_state' : example['mocap_qpos-robot0:mocap'],
      'grp_state' : None,
      'goal_state' : example['goal_qpos'],
      'obj_state' : example['obj_qpos'],
  }
  if fetch_target:
    processed_example['target_rgb'] = example['target_rgb']
    processed_example['target_depth'] = example['target_depth']

  jnt_state = tf.stack(
      values=[
          example['joint_qpos-robot0:shoulder_pan_joint'],
          example['joint_qpos-robot0:shoulder_lift_joint'],
          example['joint_qpos-robot0:upperarm_roll_joint'],
          example['joint_qpos-robot0:elbow_flex_joint'],
          example['joint_qpos-robot0:forearm_roll_joint'],
          example['joint_qpos-robot0:wrist_flex_joint'],
          example['joint_qpos-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_jnt_state')
  processed_example['jnt_state'] = jnt_state
  vel_state = tf.stack(
      values=[
          example['joint_qvel-robot0:shoulder_pan_joint'],
          example['joint_qvel-robot0:shoulder_lift_joint'],
          example['joint_qvel-robot0:upperarm_roll_joint'],
          example['joint_qvel-robot0:elbow_flex_joint'],
          example['joint_qvel-robot0:forearm_roll_joint'],
          example['joint_qvel-robot0:wrist_flex_joint'],
          example['joint_qvel-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_vel_state')
  processed_example['vel_state'] = vel_state
  grp_state = tf.stack(
      values=[
          example['joint_qpos-robot0:l_gripper_finger_joint'],
          example['joint_qpos-robot0:r_gripper_finger_joint'],
      ],
      axis=1,
      name='stack_grp_state')
  processed_example['grp_state'] = grp_state
  return processed_example

def _preprocess_targets_v3(example):
  """Append command vectors (next frame states)."""
  # assign shifted vel, grp, ee states as targets
  vel_target = tf.roll(example['vel_state'], shift=-1, axis=0)
  ee_target = tf.roll(example['ee_state'], shift=-1, axis=0)
  grp_target = tf.roll(example['grp_state'], shift=-1, axis=0)
  example['vel_target'] = vel_target
  example['ee_target'] = ee_target
  example['grp_target'] = grp_target
  # globally drop last frame
  for k in example.keys():
    if k in ['target_rgb', 'target_depth']:  # leave target frames untouched
      pass
    else:  # drop last frame in sequences
      example[k] = example[k][:-1]
  return example

def _window_v3(example, meta, window_size=4):
  """Preprocess a sequence into sliding windows."""
  seq_length = meta.episode_length - 1  # must be set as literal here to enforce shape inference!
  num_windows = seq_length-window_size+1
  for k in example.keys():
    seq_tensor = example[k]
    if k == 'step':
      window = tf.map_fn(
          lambda i: seq_tensor[i:i+window_size], tf.range(num_windows), dtype=tf.int64)
    elif k in ['target_rgb', 'target_depth']:  # tile targets to add to every window
      window = tf.tile(seq_tensor, [num_windows, 1, 1, 1])
    else:
      window = tf.map_fn(
          lambda i: seq_tensor[i:i+window_size], tf.range(num_windows), dtype=tf.float32)
    example[k] = window
    # print(k, example[k])  # DEBUG
  return example

def _prepare_v3(window_example, fetch_target):
  """Transforms each example window into a (feature, label) tuple."""
  feature = {
      'step' : window_example['step'],  # [K]
      'ts' : window_example['ts'],  # [K]
      'rgb' : window_example['rgb'],  # [K, IMG_H, IMG_W, 3]
      'depth' : window_example['depth'],  # [K, IMG_H, IMG_W, 1]
      'cmd' : window_example['cmd'],  # [K, 2]: cmd for each finger actuator
      'jnt_state' : window_example['jnt_state'],  # [K, 7]: pos of all 7 joints of arm
      'vel_state' : window_example['vel_state'],  # [K, 7]: vel of all 7 joints of arm
      'ee_state' : window_example['ee_state'],  # [K, 7]: qpos of EE
      'grp_state' : window_example['grp_state'],  # [K, 2]: pos of each finger
      'goal_state' : window_example['goal_state'],  # [K, 7]: qpos of target object
      'obj_state' : window_example['obj_state'],  # [K, 7]: qpos of moved object
  }
  if fetch_target:
    feature['target_rgb'] = window_example['target_rgb']  # [IMG_H, IMG_W, 3]
    feature['target_depth'] = window_example['target_depth']  # [IMG_H, IMG_W, 1]
  label = {
      'vel_target' : window_example['vel_target'][-1],  # [7]: target velocities for all 7 joints of arm
      'ee_target' : window_example['ee_target'][-1],  # [7]: target qpos for EE
      'grp_target' : window_example['grp_target'][-1],  # [2]: target positions for fingers
  }
  return (feature, label)

def pickplace_input_fn_v3(
    dataset_dir,
    split_name,
    mode,  # train | val | test
    window_size=4,  # length of fetched (sub-)sequences, max. episode length (=100)
    fetch_target=False,  # also fetch target images, 20% performance drop!
    shuffle_buffer=128,  # number of elements to shuffle next batch from
    batch_size=1,
    num_epochs=1,
    num_threads=4,  # number of parallel threads for data preparation
    prefetch_size=4,  # number of batches to prefetch into the pipeline
    seed=None):
  """
  Input_fn for a dataset of tfrecords with `PickAndPlaceEncodingV2`.
  Assumed dataset structure:
  <dataset_dir>/
    meta/
      meta_info.json
      init-*.csv
    data/
      *.tfrecord.zlib
      *.pkl
    splits/
      <split_name>/
        train.txt
        eval.txt
        test.txt
        init-train.csv
        init-eval.csv
        init-test.csv
  """
  # collect tfrecords
  meta = _get_meta_v2(dataset_dir)
  # print(meta)
  tfrecord_paths = _collect_tfrecords_v2(dataset_dir, split_name, mode)
  # print(tfrecord_paths)
  num_tfrecords = len(tfrecord_paths)
  print("#tfrecords: %d" % num_tfrecords)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(  # set of proto strings
      filenames=tfrecord_paths,
      compression_type='ZLIB',  # TODO: infer compression type from tfrecords
      num_parallel_reads=num_threads)
  # shuffle data for training
  if mode == 'train':  # shuffle on sequence-level
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
  # parse tfrecords
  dataset = dataset.map(
      lambda proto_example: _parse_v3(proto_example, fetch_target, meta),
      num_parallel_calls=num_threads)
  # transform state vectors
  dataset = dataset.map(
      lambda example: _preprocess_states_v3(example, fetch_target),
      num_parallel_calls=num_threads)
  # transform command vectors
  dataset = dataset.map(
      lambda example: _preprocess_targets_v3(example),
      num_parallel_calls=num_threads)
  # create sliding windows
  dataset = dataset.map(
      lambda example: _window_v3(example, meta, window_size=window_size),
      num_parallel_calls=num_threads)
  dataset = dataset.apply(tf.data.experimental.unbatch())  # unbatch frame windows
  # prepare data as supervised tuples: (feature, label)
  dataset = dataset.map(
      lambda window_example: _prepare_v3(window_example, fetch_target),
      num_parallel_calls=num_threads)
  # set up batching
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(prefetch_size)
  return dataset

def pickplace_dataset_v3(dataset_dir, num_threads=4):
  """
  Returns a dataset of tfrecords with `PickAndPlaceEncodingV3`.
  Assumed dataset structure:
  <dataset_dir>/
    meta/
      meta_info.json
      init-*.csv
    data/
      *.tfrecord.zlib
      *.pkl
    splits/
      <split_name>/
        train.txt
        eval.txt
        test.txt
        init-train.csv
        init-eval.csv
        init-test.csv
  """
  # collect tfrecords
  meta = _get_meta_v2(dataset_dir)
  # print(meta)
  tfrecord_paths = _collect_tfrecords_v2(dataset_dir, split_name=None, mode=None)
  # print(tfrecord_paths)
  num_tfrecords = len(tfrecord_paths)
  print("#tfrecords: %d" % num_tfrecords)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(  # set of proto strings
      filenames=tfrecord_paths,
      compression_type='ZLIB',  # TODO: infer compression type from tfrecords
      num_parallel_reads=num_threads)
  # parse tfrecords
  dataset = dataset.map(
      lambda proto_example: _parse_v3(proto_example, fetch_target=False, meta=meta),
      num_parallel_calls=num_threads)
  return dataset


# ---------- input_fn V2 ----------

def _get_meta_v2(data_dir):
  """Reads the meta information of the dataset."""
  meta_info_path = os.path.join(data_dir, 'meta', 'meta_info.json')
  with open(meta_info_path, 'r') as fp:
    meta_info_dict = json.load(fp)
  meta = PickAndPlaceMetaV2(**meta_info_dict)
  return meta

def _collect_tfrecords_v2(data_dir, split_name, mode):
  """
  Collects the paths to all tfrecords in the dataset.
  If split_name and mode are None, returns all tfrecords of the dataset.
  """
  record_dir = os.path.join(data_dir, 'data')
  if split_name is None and mode is None:  # TODO: document special case: retrieve entire dataset!
    record_files = [fn for fn in os.listdir(record_dir) if fn.endswith('.tfrecord.zlib')]
  else:  # standard case: collect data for a particular split
    split_file = os.path.join(data_dir, 'splits', split_name, '%s.txt' % (mode, ))
    with open(split_file) as fp:
      record_files = fp.read().split('\n')
  tfrecord_paths = [os.path.join(record_dir, fn) for fn in record_files if fn.endswith('.tfrecord.zlib')]
  return tfrecord_paths

def _parse_v2(proto_example, meta):
  """Decode frame sequence from tfrecord and normalize values."""
  # decoder setup
  encoding = PickAndPlaceEncodingV2(meta)
  context_decoder, sequence_decoder = encoding.decode()
  # parse proto example
  context_data, sequence_data = tf.parse_single_sequence_example(
      serialized=proto_example,
      context_features=context_decoder,
      sequence_features=sequence_decoder)
  parsed_example = {}
  # parsed_example.update(context_data)  # context == meta here!
  parsed_example.update(sequence_data)
  # reshape data fields
  rgb = parsed_example['rgb']
  parsed_example['rgb'] = tf.reshape(rgb, [-1, meta.img_height, meta.img_width, 3])
  depth = parsed_example['depth']
  parsed_example['depth'] = tf.reshape(depth, [-1, meta.img_height, meta.img_width, 1])
  # normalize data
  parsed_example['rgb'] /= 255.0  # RGB recorded as uint8 [0 .. 255]
  return parsed_example

def _preprocess_states_v2(example):
  """Concatenate the frame annotations into state vectors."""
  processed_example = {
      'step' : example['step'],
      'ts' : example['ts'],
      'rgb' : example['rgb'],
      'depth' : example['depth'],
      'cmd' : example['cmd'],
      'jnt_state' : None,
      'vel_state' : None,
      'ee_state' : example['mocap_qpos-robot0:mocap'],
      'grp_state' : None,
      'obj_state' : example['object_qpos-object0:joint'],
  }
  jnt_state = tf.stack(
      values=[
          example['joint_qpos-robot0:shoulder_pan_joint'],
          example['joint_qpos-robot0:shoulder_lift_joint'],
          example['joint_qpos-robot0:upperarm_roll_joint'],
          example['joint_qpos-robot0:elbow_flex_joint'],
          example['joint_qpos-robot0:forearm_roll_joint'],
          example['joint_qpos-robot0:wrist_flex_joint'],
          example['joint_qpos-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_jnt_state')
  processed_example['jnt_state'] = jnt_state
  vel_state = tf.stack(
      values=[
          example['joint_qvel-robot0:shoulder_pan_joint'],
          example['joint_qvel-robot0:shoulder_lift_joint'],
          example['joint_qvel-robot0:upperarm_roll_joint'],
          example['joint_qvel-robot0:elbow_flex_joint'],
          example['joint_qvel-robot0:forearm_roll_joint'],
          example['joint_qvel-robot0:wrist_flex_joint'],
          example['joint_qvel-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_vel_state')
  processed_example['vel_state'] = vel_state
  grp_state = tf.stack(
      values=[
          example['joint_qpos-robot0:l_gripper_finger_joint'],
          example['joint_qpos-robot0:r_gripper_finger_joint'],
      ],
      axis=1,
      name='stack_grp_state')
  processed_example['grp_state'] = grp_state
  return processed_example

def _preprocess_targets_v2(example):
  """Append command vectors (next frame states)."""
  # assign shifted vel, grp, ee states as targets
  vel_target = tf.roll(example['vel_state'], shift=-1, axis=0)
  ee_target = tf.roll(example['ee_state'], shift=-1, axis=0)
  grp_target = tf.roll(example['grp_state'], shift=-1, axis=0)
  example['vel_target'] = vel_target
  example['ee_target'] = ee_target
  example['grp_target'] = grp_target
  # globally drop last frame
  for k in example.keys():
    example[k] = example[k][:-1]
  return example

def _window_v2(example, meta, window_size=4):
  """Preprocess a sequence into sliding windows."""
  seq_length = meta.episode_length - 1  # must be set as literal here to enforce shape inference!
  for k in example.keys():
    seq_tensor = example[k]
    if k == 'step':
      window = tf.map_fn(
          lambda i: seq_tensor[i:i+window_size], tf.range(seq_length-window_size+1), dtype=tf.int64)
    else:
      window = tf.map_fn(
          lambda i: seq_tensor[i:i+window_size], tf.range(seq_length-window_size+1), dtype=tf.float32)
    example[k] = window
  return example

def _prepare_v2(window_example):
  """Transforms each example window into a (feature, label) tuple."""
  feature = {
      'step' : window_example['step'],  # [K]
      'ts' : window_example['ts'],  # [K]
      'rgb' : window_example['rgb'],  # [K, IMG_H, IMG_W, 3]
      'depth' : window_example['depth'],  # [K, IMG_H, IMG_W, 1]
      'cmd' : window_example['cmd'],  # [K, 2]: cmd for each finger actuator
      'jnt_state' : window_example['jnt_state'],  # [K, 7]: pos of all 7 joints of arm
      'vel_state' : window_example['vel_state'],  # [K, 7]: vel of all 7 joints of arm
      'ee_state' : window_example['ee_state'],  # [K, 7]: qpos of EE
      'grp_state' : window_example['grp_state'],  # [K, 2]: pos of each finger
      'obj_state' : window_example['obj_state'],  # [K, 7]: qpos of moved object
  }
  label = {
      'vel_target' : window_example['vel_target'][-1],  # [7]: target velocities for all 7 joints of arm
      'ee_target' : window_example['ee_target'][-1],  # [7]: target qpos for EE
      'grp_target' : window_example['grp_target'][-1],  # [2]: target positions for fingers
  }
  return (feature, label)

def pickplace_input_fn_v2(
    dataset_dir,
    split_name,
    mode,  # train | val | test
    window_size=4,  # length of fetched (sub-)sequences, max. episode length (=100)
    shuffle_buffer=128,  # number of elements to shuffle next batch from
    batch_size=1,
    num_epochs=1,
    num_threads=4,  # number of parallel threads for data preparation
    prefetch_size=4,  # number of batches to prefetch into the pipeline
    seed=None):
  """
  Input_fn for a dataset of tfrecords with `PickAndPlaceEncodingV2`.
  Assumed dataset structure:
  <dataset_dir>/
    meta/
      meta_info.json
      init-*.csv
    data/
      *.tfrecord.zlib
      *.pkl
    splits/
      <split_name>/
        train.txt
        eval.txt
        test.txt
        init-train.csv
        init-eval.csv
        init-test.csv
  """
  # collect tfrecords
  meta = _get_meta_v2(dataset_dir)
  # print(meta)
  tfrecord_paths = _collect_tfrecords_v2(dataset_dir, split_name, mode)
  # print(tfrecord_paths)
  num_tfrecords = len(tfrecord_paths)
  print("#tfrecords: %d" % num_tfrecords)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(  # set of proto strings
      filenames=tfrecord_paths,
      compression_type='ZLIB',  # TODO: infer compression type from tfrecords
      num_parallel_reads=num_threads)
  # shuffle data for training
  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
  # parse tfrecords
  dataset = dataset.map(
      lambda proto_example: _parse_v2(proto_example, meta),
      num_parallel_calls=num_threads)
  # transform state vectors
  dataset = dataset.map(
      lambda example: _preprocess_states_v2(example),
      num_parallel_calls=num_threads)
  # transform command vectors
  dataset = dataset.map(
      lambda example: _preprocess_targets_v2(example),
      num_parallel_calls=num_threads)
  # create sliding windows
  dataset = dataset.map(
      lambda example: _window_v2(example, meta, window_size=window_size),
      num_parallel_calls=num_threads)
  dataset = dataset.apply(tf.data.experimental.unbatch())  # unbatch frame windows
  # prepare data as supervised tuples: (feature, label)
  dataset = dataset.map(
      lambda window_example: _prepare_v2(window_example),
      num_parallel_calls=num_threads)
  # set up batching
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(prefetch_size)
  return dataset


# ---------- input_fn V1 ----------

def _get_meta_v1(data_dir):
  """Reads the meta information of the dataset."""
  meta_info_path = os.path.join(data_dir, 'meta', 'meta_info.json')
  with open(meta_info_path, 'r') as fp:
    meta_info_dict = json.load(fp)
  meta = PickAndPlaceMetaV1(**meta_info_dict)
  return meta

def _collect_tfrecords_v1(data_dir, split_name, mode):
  """Collects the paths to all tfrecords in the dataset."""
  split_file = os.path.join(data_dir, 'splits', split_name, '%s.txt' % (mode, ))
  with open(split_file) as fp:
    record_files = fp.read().split('\n')[:-1]
  record_dir = os.path.join(data_dir, 'data')
  # record_files = [fn for fn in os.listdir(record_dir) if fn.endswith('.tfrecord.zlib')]
  tfrecord_paths = [os.path.join(record_dir, fn) for fn in record_files]
  return tfrecord_paths

def _parse_v1(proto_example, meta):
  """Decode frame sequence from tfrecord and normalize values."""
  # decoder setup
  encoding = PickAndPlaceEncodingV1(
      meta.monitored_joints, meta.monitored_mocaps, meta.monitored_objects)
  context_decoder, sequence_decoder = encoding.decode()
  # parse proto example
  context_data, sequence_data = tf.parse_single_sequence_example(
      serialized=proto_example,
      context_features=context_decoder,
      sequence_features=sequence_decoder)
  parsed_example = {}
  # parsed_example.update(context_data)  # context == meta here!
  parsed_example.update(sequence_data)
  # reshape data fields
  rgb = parsed_example['rgb']
  parsed_example['rgb'] = tf.reshape(rgb, [-1, _IMG_H, _IMG_W, _IMG_C])
  # normalize data
  parsed_example['rgb'] /= 255.0  # RGB recorded as uint8 [0 .. 255]
  return parsed_example

# TODO: prepend K-1 frames to each sequence (pad with first frame) to simulate cold-start
def _pad_sequence_v1(example):
  err_msg = "Sequence padding not implemented, yet!"
  raise NotImplementedError(err_msg)

def _preprocess_states_v1(example):
  """Concatenate the frame annotations into state vectors."""
  processed_example = {
      'step' : example['step'],
      'ts' : example['ts'],
      'rgb' : example['rgb'],
      'cmd' : example['cmd'],
      'jnt_state' : None,
      'vel_state' : None,
      'ee_state' : example['mocap_qpos-robot0:mocap'],
      'grp_state' : None,
      'obj_state' : example['object_qpos-object0:joint'],
  }
  jnt_state = tf.stack(
      values=[
          example['joint_qpos-robot0:shoulder_pan_joint'],
          example['joint_qpos-robot0:shoulder_lift_joint'],
          example['joint_qpos-robot0:upperarm_roll_joint'],
          example['joint_qpos-robot0:elbow_flex_joint'],
          example['joint_qpos-robot0:forearm_roll_joint'],
          example['joint_qpos-robot0:wrist_flex_joint'],
          example['joint_qpos-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_jnt_state')
  processed_example['jnt_state'] = jnt_state
  vel_state = tf.stack(
      values=[
          example['joint_qvel-robot0:shoulder_pan_joint'],
          example['joint_qvel-robot0:shoulder_lift_joint'],
          example['joint_qvel-robot0:upperarm_roll_joint'],
          example['joint_qvel-robot0:elbow_flex_joint'],
          example['joint_qvel-robot0:forearm_roll_joint'],
          example['joint_qvel-robot0:wrist_flex_joint'],
          example['joint_qvel-robot0:wrist_roll_joint'],
      ],
      axis=1,
      name='stack_vel_state')
  processed_example['vel_state'] = vel_state
  grp_state = tf.stack(
      values=[
          example['joint_qpos-robot0:l_gripper_finger_joint'],
          example['joint_qpos-robot0:r_gripper_finger_joint'],
      ],
      axis=1,
      name='stack_grp_state')
  processed_example['grp_state'] = grp_state
  return processed_example

def _preprocess_targets_v1(example):
  """Append command vectors (next frame states)."""
  # assign shifted vel, grp, ee states as targets
  vel_target = tf.roll(example['vel_state'], shift=-1, axis=0)
  ee_target = tf.roll(example['ee_state'], shift=-1, axis=0)
  grp_target = tf.roll(example['grp_state'], shift=-1, axis=0)
  example['vel_target'] = vel_target
  example['ee_target'] = ee_target
  example['grp_target'] = grp_target
  # globally drop last frame
  for k in example.keys():
    example[k] = example[k][:-1]
  return example

def _window_v1(example, seq_length=99, window_size=4):
  """Preprocess a sequence into sliding windows."""
  # TODO: pass episode_length from meta struct; seq_length = episode_length - 1
  seq_length = 99  # must be set as literal here for shape inference engine!
  for k in example.keys():
    seq_tensor = example[k]
    if k == 'step':
      window = tf.map_fn(
          lambda i: seq_tensor[i:i+window_size], tf.range(seq_length-window_size+1), dtype=tf.int64)
    else:
      window = tf.map_fn(
          lambda i: seq_tensor[i:i+window_size], tf.range(seq_length-window_size+1), dtype=tf.float32)
    example[k] = window
  return example

def _prepare_v1(window_example):
  """Transforms each example window into a (feature, label) tuple."""
  feature = {
      'step' : window_example['step'],  # [K]
      'ts' : window_example['ts'],  # [K]
      'rgb' : window_example['rgb'],  # [K, _IMG_H, _IMG_W, _IMG_C]
      'cmd' : window_example['cmd'],  # [K, 2]: cmd for each finger actuator
      'jnt_state' : window_example['jnt_state'],  # [K, 7]: pos of all 7 joints of arm
      'vel_state' : window_example['vel_state'],  # [K, 7]: vel of all 7 joints of arm
      'ee_state' : window_example['ee_state'],  # [K, 7]: qpos of EE
      'grp_state' : window_example['grp_state'],  # [K, 2]: pos of each finger
      'obj_state' : window_example['obj_state'],  # [K, 7]: qpos of moved object
  }
  label = {
      'vel_target' : window_example['vel_target'][-1],  # [7]: target velocities for all 7 joints of arm
      'ee_target' : window_example['ee_target'][-1],  # [7]: target qpos for EE
      'grp_target' : window_example['grp_target'][-1],  # [2]: target positions for fingers
  }
  return (feature, label)

def pickplace_input_fn_v1(
    dataset_dir,
    split_name,
    mode,  # train | val | test
    window_size=4,  # length of fetched (sub-)sequences, max. episode length (=100)
    shuffle_buffer=128,  # number of elements to shuffle next batch from
    batch_size=1,
    num_epochs=1,
    num_threads=4,  # number of parallel threads for data preparation
    prefetch_size=4,  # number of batches to prefetch into the pipeline
    seed=None):
  """
  Input_fn for a dataset of tfrecords with `PickAndPlaceEncodingV1`.
  Assumed dataset structure:
  <data_dir>/
    meta_info.json
    train/
    val/
    test/
  """
  # collect tfrecords
  meta = _get_meta_v1(dataset_dir)
  # print(meta)
  tfrecord_paths = _collect_tfrecords_v1(dataset_dir, split_name, mode)
  # print(tfrecord_paths)
  num_tfrecords = len(tfrecord_paths)
  print("#tfrecords: %d" % num_tfrecords)
  # create TFRecordDataset
  dataset = tf.data.TFRecordDataset(  # set of proto strings
      filenames=tfrecord_paths,
      compression_type='ZLIB',  # TODO: infer compression type from tfrecords
      num_parallel_reads=num_threads)
  # shuffle data for training
  if mode == 'train':
    dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)
  # parse tfrecords
  dataset = dataset.map(
      lambda proto_example: _parse_v1(proto_example, meta),
      num_parallel_calls=num_threads)
  # transform state vectors
  dataset = dataset.map(
      lambda example: _preprocess_states_v1(example),
      num_parallel_calls=num_threads)
  # transform command vectors
  dataset = dataset.map(
      lambda example: _preprocess_targets_v1(example),
      num_parallel_calls=num_threads)
  # create sliding windows
  dataset = dataset.map(
      lambda example: _window_v1(example, window_size=window_size),
      num_parallel_calls=num_threads)
  dataset = dataset.apply(tf.data.experimental.unbatch())  # unbatch frame windows
  # prepare data as supervised tuples: (feature, label)
  dataset = dataset.map(
      lambda window_example: _prepare_v1(window_example),
      num_parallel_calls=num_threads)
  # set up batching
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(prefetch_size)
  return dataset


# ---------- register input functions ----------

_ENCODING2INPUTFN = {
    'v1' : pickplace_input_fn_v1,
    'v2' : pickplace_input_fn_v2,
    'v3' : pickplace_input_fn_v3,
    'v4' : pickplace_input_fn_v4,
}
