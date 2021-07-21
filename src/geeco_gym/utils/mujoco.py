"""
Contains various utility functions to work with MjSim objects.
"""

from functools import lru_cache

from mujoco_py import MjSim


class MjSimHelper(object):

  def __init__(self, sim: MjSim):
    """A class wrapping an instance of `MjSim` providing various helper functions."""
    self._mjsim = sim
    self._mjmodel = self._mjsim.model

  @lru_cache(maxsize=None)
  def get_body_geom_ids(self, body_name: str):
    """Returns a list of all geom IDs associated with the given body."""
    mj_body_id = self._mjmodel.body_name2id(body_name)
    first_geom_id = self._mjmodel.body_geomadr[mj_body_id]
    range_geom_id = self._mjmodel.body_geomnum[mj_body_id]
    return list(range(first_geom_id, first_geom_id + range_geom_id))

  @lru_cache(maxsize=None)
  def get_body_geom_names(self, body_name: str):
    """Returns a list of all geom names associated with the given body."""
    geom_ids = self.get_body_geom_ids(body_name)
    return [self._mjmodel.geom_id2name(idx) for idx in geom_ids]

  def show_geom(self, geom_id: int):
    """Makes a geom fully visible."""
    self._mjmodel.geom_rgba[geom_id][3] = 1.0

  def hide_geom(self, geom_id: int):
    """Makes a geom invsible."""
    self._mjmodel.geom_rgba[geom_id][3] = 0.0

  def enable_geom_physics(self, geom_id: int):
    """
    Enables a geom by making it visible (alpha channel to 1.0) and enabling
    physical contacts with it (contype & conaffinity -> 1).
    """
    self._mjmodel.geom_contype[geom_id] = 1
    self._mjmodel.geom_conaffinity[geom_id] = 1

  def disable_geom_physics(self, geom_id: int):
    """
    Disables a geom by making it invisible (alpha channel to 0.0) and disabling
    physical contacts with it (contype & conaffinity -> 0).
    """
    self._mjmodel.geom_contype[geom_id] = 0
    self._mjmodel.geom_conaffinity[geom_id] = 0

  def enable_body(self, body_name: str):
    """Enables the body by enabling all of its associated geoms."""
    geom_ids = self.get_body_geom_ids(body_name)
    for geom_id in geom_ids:
      self.show_geom(geom_id)
      self.enable_geom_physics(geom_id)

  def disable_body(self, body_name: str):
    """Disables the body by disabling all of its associated geoms."""
    geom_ids = self.get_body_geom_ids(body_name)
    for geom_id in geom_ids:
      self.hide_geom(geom_id)
      self.disable_geom_physics(geom_id)

  def enable_body_physics(self, body_name: str):
    """Enables the body for physical interaction."""
    geom_ids = self.get_body_geom_ids(body_name)
    for geom_id in geom_ids:
      self.enable_geom_physics(geom_id)

  def disable_body_physics(self, body_name: str):
    """Disables the body for physical interaction."""
    geom_ids = self.get_body_geom_ids(body_name)
    for geom_id in geom_ids:
      self.disable_geom_physics(geom_id)
