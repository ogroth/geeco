"""Contains utilies for matplotlib."""

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


def create_image_grid(
    num_examples, max_cols_per_row=8, tile_height=4.0, tile_width=4.0, tile_pad=0.8):
  """Defines an image grid for `num_examples` and loads the plot to be released by plt.show()"""
  ncols = int(np.min([num_examples, max_cols_per_row]))
  nrows = int(np.ceil(num_examples / max_cols_per_row))
  fig = plt.figure(figsize=(tile_width * ncols, tile_height * nrows))
  grid = ImageGrid(
      fig, 111,  # similar to subplot(111)
      nrows_ncols=(nrows, ncols),
      axes_pad=tile_pad,  # pad between axes in inch.
  )
  return grid
