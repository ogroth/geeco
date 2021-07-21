"""Utilities for video handling."""

from moviepy.editor import VideoFileClip


class VideoCycler:

  def __init__(self, video_path):
    """Loads the video from `video_path` and cycles it using a memory-economic generator."""
    self.video = VideoFileClip(video_path)
    self.reset()
        
  def reset(self):
    self.ite = self.video.iter_frames()
        
  def __next__(self):
    try:
      next_frame = next(self.ite)
    except StopIteration:
      self.reset()
      next_frame = next(self.ite)
    return next_frame
    
  def __iter__(self):
    return self
