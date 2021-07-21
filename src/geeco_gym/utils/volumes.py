"""Helper functions for volumes."""

import numpy as np

def sample_point_within_sphere(radius=1.0):
  """
  Randomly samples a point in a unit sphere and returns (x, y, z).
  Source: https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
  """
    # var u = Math.random();
    # var x1 = randn();
    # var x2 = randn();
    # var x3 = randn();

    # var mag = Math.sqrt(x1*x1 + x2*x2 + x3*x3);
    # x1 /= mag; x2 /= mag; x3 /= mag;

    # // Math.cbrt is cube root
    # var c = Math.cbrt(u);

    # return {x: x1*c, y:x2*c, z:x3*c};
  u = np.random.uniform(low=0.0, high=radius)
  x1 = np.random.normal(loc=0.0, scale=radius)
  x2 = np.random.normal(loc=0.0, scale=radius)
  x3 = np.random.normal(loc=0.0, scale=radius)
  mag = np.sqrt(x1*x1 + x2*x2 + x3*x3)
  x1 /= mag
  x2 /= mag
  x3 /= mag
  c = np.cbrt(u)
  return (x1*c, x2*c, x3*c)
