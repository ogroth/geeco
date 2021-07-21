"""Contains utility functions to work on grids."""

import numpy as np


# @deprecated!
def compute_grid(minmax_x, minmax_y, tiling_xy):
  """Creates a 2D grid with the given tiling."""
  length_x = minmax_x[1] - minmax_x[0]
  length_y = minmax_y[1] - minmax_y[0]
  step_x = length_x / tiling_xy[0]
  step_y = length_y / tiling_xy[1]
  borders_x = list(np.arange(minmax_x[0], minmax_x[1] + step_x, step_x))
  borders_y = list(np.arange(minmax_y[0], minmax_y[1] + step_y, step_y))
  centers_x = list(map(lambda a, b: np.mean([a, b]), borders_x, borders_x[1:]))
  centers_y = list(map(lambda a, b: np.mean([a, b]), borders_y, borders_y[1:]))
  centers = []
  for cx in centers_x:
    for cy in centers_y:
      centers.append((cx, cy))
  return centers

def compute_grid_v2(minmax_x, minmax_y, tiling_xy):
  """Creates a 2D grid with the given tiling."""
  # tile*2+1 to compute centers (equal space left and right) and exclude borders
  centers_x = np.linspace(minmax_x[0], minmax_x[1], tiling_xy[0]*2+1)
  centers_y = np.linspace(minmax_y[0], minmax_y[1], tiling_xy[1]*2+1)
  index_x = list(range(1, tiling_xy[0]*2+1, 2))
  index_y = list(range(1, tiling_xy[1]*2+1, 2))
  centers_x = list(centers_x[index_x])
  centers_y = list(centers_y[index_y])
  centers = []
  for cx in centers_x:
    for cy in centers_y:
      centers.append((cx, cy))
  return centers

def sample_spawn_grid(num_points, minmax_x, minmax_y, tiling_xy):
  """
  Creates a 2D grid with the given tiling. Then samples `num_points` grid cells
  and returns their respective centers.
  """
  grid_centers = compute_grid_v2(minmax_x, minmax_y, tiling_xy)
  idx = list(np.random.choice(len(grid_centers), num_points, replace=False))
  spawn_points = [grid_centers[i] for i in idx]
  return spawn_points
