"""
Defines recorder to gather data from GEECO simulations and experiments.
"""

import os
import time
import datetime

import tensorflow as tf

from .utils.tfrecord import convert_to_feature


# ---------- base data encodings ----------

class TfrSequenceEncoding(object):
  """
  A base class providing encoding and decoding APIs for
  tf.train.SequenceExample.
  """

  def __init__(self):
    """
    Initializes a TfrSequenceEncoding by defining the data keys.
    """
    self._context_keys = []
    self._frame_keys = []

  @property
  def context_keys(self):
    return self._context_keys

  @property
  def frame_keys(self):
    return self._frame_keys

  def encode(self, data: dict) -> tf.train.SequenceExample:
    """Encodes a data dictionary into a tf.train.SequenceExample."""
    # meta information / sequence context
    context_features = {}
    for k in self._context_keys:
      context_features[k] = convert_to_feature(data[k])
    context_encoding = tf.train.Features(feature=context_features)
    # convert data frames into lists of tf.train.Feature
    frame_feature_lists = dict([(k, []) for k in self._frame_keys])
    for frame in data['sequence']:
      for k in self._frame_keys:
        feat = convert_to_feature(frame[k])
        frame_feature_lists[k].append(feat)
    # build frame encoding
    feature_list = dict(
        [(k, tf.train.FeatureList(feature=frame_feature_lists[k])) \
          for k in self._frame_keys]
    )
    feature_encoding = tf.train.FeatureLists(feature_list=feature_list)
    # assemble SequenceExample
    example = tf.train.SequenceExample(
        context=context_encoding, feature_lists=feature_encoding)
    return example

  def decode(self):
    """
    Returns a 'context_features' and 'sequence_features' dictionary, specifying
    how the protobuf data is interpreted in tensorflow.
    """
    raise NotImplementedError()


# ---------- data recorders ----------

class TfrSequenceRecorder(object):
  """
  A data recorder which writes sequences of structured data frames (key-value
  lists) to tfrecord protobuf files.
  """

  def __init__(
      self,
      encoding: TfrSequenceEncoding,
      sequence_context: dict,
      record_dir, record_name: str = None):
    """
    Initializes an TfrSequenceRecorder capable of encoding structured data
    frames into tfrecords.

    Args:
      sequence_context (dict):
      encoding (TfrSequenceEncoding):
    """
    self._encoding = encoding
    self._sequence_context = sequence_context
    self._sequence_frames = []
    self._record_dir = record_dir
    if record_name is None:
      ts = time.time()
      record_name = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    self._record_name = record_name
    self._record_path = self._get_record_path()

  @property
  def record_name(self):
    '''Returns the name of the tfrecord file this recorder writes to.'''
    return self._record_name

  @property
  def record_path(self):
    '''Returns the path of the tfrecord file this recorder writes to.'''
    return self._record_path

  def _get_record_path(self):
    fn = "%s.tfrecord" % (self._record_name, )
    record_path = os.path.join(self._record_dir, fn)
    return record_path

  def _has_valid_format(self, frame):
    expected_keys = set(self._encoding.frame_keys)
    given_keys = set(frame.keys())
    return given_keys.issubset(expected_keys) and expected_keys.issubset(given_keys)

  def feed(self, frame):
    """
    Appends a data frame to the sequence buffer.
    """
    if self._has_valid_format(frame):
      self._sequence_frames.append(frame)
    else:
      expected_keys = set(self._encoding.frame_keys)
      given_keys = set(frame.keys())
      err_msg = "The given frame does not match the expected data fields!\n\
          Given data fields: %s\n\
          Expected data fields: %s" % (given_keys, expected_keys)
      raise ValueError(err_msg)

  def finalize(self, compression='none'):
    """
    Compiles sequence context and data frames, encodes them as
    tf.train.SequenceExample and writes them to a tfrecord file.

    :params:
      compression: none | gzip | zlib
    """
    options = {
        'none' : tf.io.TFRecordCompressionType.NONE,
        'gzip' : tf.io.TFRecordCompressionType.GZIP,
        'zlib' : tf.io.TFRecordCompressionType.ZLIB,
    }
    opt = options[compression]
    data = {}
    data.update(self._sequence_context)
    data['sequence'] = self._sequence_frames
    example = self._encoding.encode(data)
    tfrecord_path = self._get_record_path()
    if compression != 'none':
      tfrecord_path = tfrecord_path + '.%s' % (compression, )
    with tf.python_io.TFRecordWriter(tfrecord_path, options=opt) as writer:
      writer.write(example.SerializeToString())
