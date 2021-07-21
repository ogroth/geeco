"""
Defines a recorder which is used in the MjEngine to record a video stream for a
given modality and camera.
"""

import os
import time
import shutil
from multiprocessing import Process

import imageio


# constants & defaults
# ----------------------------------------
_DEFAULT_RES_HEIGHT = 768
_DEFAULT_RES_WIDTH = 1024
# _DEFAULT_RES_HEIGHT = 720
# _DEFAULT_RES_WIDTH = 1280
_DEFAULT_BUFFER_SIZE = 1500
_DEFAULT_FPS = 25


# recorder classes
# ----------------------------------------

class MjRecorder(object):
  """
  Base recorder class which can be registered with the MjEngine to record data
  streams from simulations.
  """

  def __init__(
      self,
      ctx_name: str, ctx_type: str,
      record_name: str, record_dir):
    """
    Initializes a basic MjRecorder.

    Args:
      ctx_name (str): The name of the context this recorder is recording (e.g.
        'RGB_CAMERA01').
      ctx_type (str): The name of the data type this recorder is recording (e.g.
        'RGB').
      record_name (str): The name of the record file to be written.
      record_dir: The directory where the record will be written to.
    """
    # primary recorder information
    self._ctx_name = ctx_name
    self._ctx_type = ctx_type
    self._record_name = record_name
    self._record_dir = record_dir

    # maintain record batches
    self._record_paths = []
    self._batch_idx = 1

    # maintain current data frames
    self._frames = []

  @property
  def context_name(self):
    """The context name of the recorder."""
    return self._ctx_name

  @property
  def context_type(self):
    """The context type of the recorder."""
    return self._ctx_type

  def feed(self, frame):
    """Feeds a data frame to the recorder."""
    raise NotImplementedError()

  def finalize(self):
    """Concatenates all recorded batches and cleans up."""
    raise NotImplementedError()


class MjVideoRecorder(MjRecorder):
  """
  Mujoco-py video stream recorder.
  """

  def __init__(
      self,
      ctx_name: str, ctx_type: str, cam_name: str, record_name: str, record_dir,
      res_height: int = _DEFAULT_RES_HEIGHT, res_width: int = _DEFAULT_RES_WIDTH,
      buffer_size: int = _DEFAULT_BUFFER_SIZE, fps: int = _DEFAULT_FPS):
    """
    Initializes an MjVideoRecorder capable of writing content of OpenGL
    rendering contexts into videos (mp4) via ffmpeg subprocesses.

    :param ctx_name: The name of the rendering context, e.g. 'RGB_01'
    :param ctx_type: The type of the rendering context, e.g. 'RGB'
    :param record_name: The filename of the recorded video.
    :param record_dir: The directory to which the video and chunks will be saved.
    ...
    """
    super().__init__(ctx_name, ctx_type, record_name, record_dir)
    self._cam_name = cam_name
    self._res_height = res_height
    self._res_width = res_width
    self._buffer_size = buffer_size
    self._fps = fps  # TODO(ogroth): ensure feeding based on recorder.fps in MjEngine!

  @staticmethod
  def _write_video(video_path, fps, frames):
    """
    Writes a frame sequence to a video path.
    Internal function to be called in subprocess to avoid stalling the main app.
    """
    video_dir, video_fn = os.path.split(video_path)
    tmp_video_path = os.path.join(video_dir, '_wip_' + video_fn)
    writer = imageio.get_writer(uri=tmp_video_path, fps=fps)
    print("Writing frames to %s..." % (tmp_video_path, ))  # TODO: logging!
    for f in frames:
      writer.append_data(f)
    writer.close()
    os.rename(tmp_video_path, video_path)
    print("Video %s is written!" % (video_path, ))  # TODO: logging!

  @staticmethod
  def _concatenate_videos(target_video_path, fps, video_paths, timeout=1000):
    """
    Concatenates a list of videos into a single one.
    Internal function to be called in subprocess to avoid stalling the main app.
    """
    if len(video_paths) > 1:
      video_dir, video_fn = os.path.split(target_video_path)
      tmp_video_path = os.path.join(video_dir, '_wip_' + video_fn)
      writer = imageio.get_writer(uri=tmp_video_path, fps=fps)
      for video_path in video_paths:
        print("Adding %s to %s..." % (video_path, target_video_path))  # TODO: logging!
        t = 0
        while True:
          t += 1
          try:  # try to get file handle
            reader = imageio.get_reader(uri=video_path)
            break
          except OSError as err:  # file not present, yet
            # print("Part %s not ready, yet." % (video_path, ))  # TODO: logging!
            if t < timeout:
              time.sleep(1.0)
            else:
              raise err
        for f in reader:
          writer.append_data(f)
      writer.close()
      os.rename(tmp_video_path, target_video_path)
      print("Video %s is written!" % (target_video_path, ))  #TODO: logging!
    elif len(video_paths) == 1:  # if only one video exists, just copy to target
      print("Adding %s to %s..." % (video_paths[0], target_video_path))  #TODO: logging!
      video_path = video_paths[0]
      t = 0
      while True:
        t += 1
        try:  # try to get file handle
          shutil.copy(video_path, target_video_path)
          break
        except OSError as err:  # file not present, yet
          # print("Part %s not ready, yet." % (video_path, ))  # TODO: logging!
          if t < timeout:
            time.sleep(1.0)
          else:
            raise err
      print("Video %s is written!" % (target_video_path, ))  # TODO: logging!
    else:
      err_msg = "No videos to concatenate!"
      raise ValueError(err_msg)

  def _get_batch_fn(self):
    video_fn = "%s-%s-%s-%06d.mp4" % (
        self._record_name, self._ctx_name, self._cam_name, self._batch_idx)
    return video_fn

  def _get_record_fn(self):
    video_fn = "%s-%s-%s.mp4" % (
        self._record_name, self._ctx_name, self._cam_name)
    return video_fn

  @property
  def camera_name(self):
    """The name of the camera recorded."""
    return self._cam_name

  @property
  def resolution(self):
    """The resolution of the recording camera."""
    return (self._res_height, self._res_width)

  def feed(self, frame):
    """Feed an image into the video stream."""
    self._frames.append(frame)
    # write current video batch if buffer size has been exceeded
    if len(self._frames) == self._buffer_size:
      video_fn = self._get_batch_fn()
      video_path = os.path.join(self._record_dir, video_fn)
      fps = self._fps
      frames = [f for f in self._frames]
      video_writer = Process(
          target=MjVideoRecorder._write_video,
          args=(video_path, fps, frames)
      )
      video_writer.start()
      # bookkeeping of batches
      self._record_paths.append(video_path)
      self._batch_idx += 1
      self._frames = []

  def flush(self):
    """Flushes the current frame buffer and writes out a video."""
    video_fn = self._get_batch_fn()
    video_path = os.path.join(self._record_dir, video_fn)
    fps = self._fps
    frames = [f for f in self._frames]
    video_writer = Process(
        target=MjVideoRecorder._write_video,
        args=(video_path, fps, frames)
    )
    video_writer.start()
    # bookkeeping of batches
    self._record_paths.append(video_path)
    self._batch_idx += 1
    self._frames = []
    return video_path

  def finalize(self):
    """Concatenates all produced video batches into one video."""
    # flush remaining frames into new video
    video_fn = self._get_batch_fn()
    video_path = os.path.join(self._record_dir, video_fn)
    fps = self._fps
    frames = [f for f in self._frames]
    video_writer = Process(
        target=MjVideoRecorder._write_video,
        args=(video_path, fps, frames)
    )
    video_writer.start()  # fork video write process
    # bookkeeping of batches
    self._record_paths.append(video_path)
    self._batch_idx += 1
    self._frames = []
    # concatenate all videos
    target_video_fn = self._get_record_fn()
    target_video_path = os.path.join(self._record_dir, target_video_fn)
    video_paths = [p for p in self._record_paths]
    video_concat = Process(
        target=MjVideoRecorder._concatenate_videos,
        args=(target_video_path, fps, video_paths)
    )
    video_concat.start()  # fork concatenation process
