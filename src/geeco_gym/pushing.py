"""Defines gym environments for stacking tasks."""

import os
import collections
import pickle
import pprint

import numpy as np

from gym import utils as gym_utils
from gym.envs.robotics import fetch_env, rotations
from gym.envs.robotics import utils as robo_utils

from mj_engine.engine.recorder import MjVideoRecorder
from data.data_recorder import TfrSequenceRecorder
from data.geeco_gym import PickAndPlaceMetaV4, PickAndPlaceEncodingV4
from .utils import grids
from .utils.volumes import sample_point_within_sphere


# ---------- constants ----------

GEECO_ROOT = os.environ['GEECO_ROOT']
MODEL_XML_PATH = {
    'push-pad1-cube1' : os.path.abspath(
        os.path.join(GEECO_ROOT, 'assets', 'gym', 'envs', 'geeco-push-pad1-cube1.xml')),
    'push-pad1-cube2' : os.path.abspath(
        os.path.join(GEECO_ROOT, 'assets', 'gym', 'envs', 'geeco-push-pad1-cube2.xml')),
    'push-pad2-cube1' : os.path.abspath(
        os.path.join(GEECO_ROOT, 'assets', 'gym', 'envs', 'geeco-push-pad2-cube1.xml')),
    'push-pad2-cube2' : os.path.abspath(
        os.path.join(GEECO_ROOT, 'assets', 'gym', 'envs', 'geeco-push-pad2-cube2.xml')),
}
RENDERING_MODES = ['viewer', 'video', 'tfrecord']
DEBUG_SITE_VISUALS = {  # site_name -> RGBA
    'target0' : np.array([1, 0, 0, 1])
}
DEBUG_GEOM_VISUALS = {  # site_name -> RGBA
    'robot0:ee_crosshair_c' : np.array([0, 0.5, 0, 0.7]),
    'robot0:ee_crosshair_x' : np.array([0, 0.5, 0, 0.1]),
    'robot0:ee_crosshair_y' : np.array([0, 0.5, 0, 0.1]),
    'robot0:ee_crosshair_z' : np.array([0, 0.5, 0, 0.1]),
}


# ---------- helper functions ----------

def _print_joint_states(sim, joint_names):
  row_format = "{:40}{:15}{:15}"
  print(row_format.format("JNT_NAME", "QPOS", "QVEL"))
  for jnt_name in joint_names:
    qpos = sim.data.get_joint_qpos(jnt_name)
    qvel = sim.data.get_joint_qvel(jnt_name)
    print(row_format.format(jnt_name, "%.04f" % qpos, "%.04f" % qvel))


# ---------- gym envs ----------

class PushingEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
  """Adapted gym.envs.robotics:FetchEnv for pushing."""

  def __init__(
      self,
      work_dir,
      shapes='push-pad1-cube1', reward_type='sparse',
      rendering_mode='viewer', frame_res=(256, 256)):
    """
    :param shapes:
    :param rendering_mode: viewer | video | tfrecord
    """
    # --- init parent FetchEnv
    initial_qpos = {
        'robot0:slide0' : 0.405,
        'robot0:slide1' : 0.48,
        'robot0:slide2' : 0.0,
    }
    model_xml_path = MODEL_XML_PATH[shapes]
    fetch_env.FetchEnv.__init__(
        self, model_xml_path, has_object=True, block_gripper=False, n_substeps=20,
        gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
        obj_range=0.15, target_range=0.15, distance_threshold=0.05,
        initial_qpos=initial_qpos, reward_type=reward_type)
    self.shapes = shapes
    # --- extended initialization
    self.reset_queue = collections.deque([])
    # --- custom class members
    self.ts = 0  # count the elapsed timesteps in the environment
    self.work_dir = work_dir  # directory to store outputs in
    self.task_goal = None  # name of goal object in pick&place task
    self.task_object = None  # name of manipulated object in pick&place task
    self.target_frame = None  # RGB(D) target frame depicting desired outcome
    # --- rendering
    self.rendering_mode = rendering_mode
    self._init_rendering()
    # --- video recording
    self.frame_res = frame_res  # (height, width)
    _h, _w = frame_res
    self.recorder = MjVideoRecorder(
        ctx_name='default', ctx_type='rgb', cam_name='cam_default',
        record_name='test', record_dir=work_dir,
        res_height=_h, res_width=_w)
    # --- replay meta: monitored joints, mocaps, actuators, objects
    self.monitored_joints = [
        'robot0:slide0',
        'robot0:slide1',
        'robot0:slide2',
        'robot0:torso_lift_joint',
        'robot0:head_pan_joint',
        'robot0:head_tilt_joint',
        'robot0:shoulder_pan_joint',
        'robot0:shoulder_lift_joint',
        'robot0:upperarm_roll_joint',
        'robot0:elbow_flex_joint',
        'robot0:forearm_roll_joint',
        'robot0:wrist_flex_joint',
        'robot0:wrist_roll_joint',
        'robot0:r_gripper_finger_joint',
        'robot0:l_gripper_finger_joint',
    ]
    self.monitored_mocaps = ['robot0:mocap']
    self.actuated_joints = [
        'robot0:l_gripper_finger_joint',
        'robot0:r_gripper_finger_joint',
    ]
    self.monitored_objects = [
        jnt_name for jnt_name in self.sim.model.joint_names \
        if jnt_name.startswith('object') or jnt_name.startswith('goal')
    ]
    # --- pkl replay buffers
    self.step_buffer = []  # frame IDs
    self.time_elapsed = []  # timestamps [s]
    self.rgb_buffer = []  # RGB frames
    self.joint_qpos_buffer = {}  # jnt_name -> qpos_t; proprioception
    self.joint_qvel_buffer = {}  # jnt_name -> velocity_t; velocity control
    self.mocap_qpos_buffer = {}  # mcp_name -> qpos_t
    self.cmd_buffer = []  # records actions given
    self.ctrl_buffer = []  # records ctrl array
    self.object_qpos_buffer = {}  # jnt_name -> qpos_t
    self._reset_replay_buffer()
    # --- tfrecorder
    meta_info_dict = {
        'episode_length' : 100,  # TODO: make dynamic!
        'img_height' : self.frame_res[0],
        'img_width' : self.frame_res[0],
        'monitored_joints' : self.monitored_joints,
        'actuated_joints' : self.actuated_joints,
        'monitored_mocaps' : self.monitored_mocaps,
        'monitored_objects' : self.monitored_objects,
        'dim_cmd' : 4,  # cartesian EE (x, y, z), discrete gripper
        'dim_ctrl' : len(self.actuated_joints),
    }
    meta = PickAndPlaceMetaV4(**meta_info_dict)
    encoding = PickAndPlaceEncodingV4(meta)
    self.meta_info_dict = meta_info_dict
    self.encoding_meta = meta
    self.tfrecorder_fn = lambda sequence_context, record_name: \
      TfrSequenceRecorder(
          encoding=encoding, sequence_context=sequence_context,
          record_dir=work_dir, record_name=record_name)
    self.tfrecorder = None  # will be set with self.start_tfrecorder()
    # --- serialize final env
    gym_utils.EzPickle.__init__(self)

  # --- properties

  def get_ts(self):
    return self.ts

  # --- replay buffer

  def _reset_replay_buffer(self):
    """
    Resets the environment's replay buffer.
    Called by PushingEnv.reset().
    Called by PushingEnv() constructor.
    """
    self.step_buffer = []
    self.time_elapsed = []
    self.rgb_buffer = []
    for jnt_name in self.monitored_joints:
      self.joint_qpos_buffer[jnt_name] = []
      self.joint_qvel_buffer[jnt_name] = []
    for mcp_name in self.monitored_mocaps:
      self.mocap_qpos_buffer[mcp_name] = []
    self.cmd_buffer = []
    self.ctrl_buffer = []
    for jnt_name in self.monitored_objects:
      self.object_qpos_buffer[jnt_name] = []

  def _append_replay_buffer(self, action):
    """
    Appends the current commands to the replay buffers of the monitored joints.
    Called by PushingEnv.step()
    """
    self.step_buffer.append(self.ts)
    self.time_elapsed.append(self.sim.data.time)
    # self.rgb_buffer.append()  # TODO: append frames only in offscreen mode!
    self.ctrl_buffer.append(np.copy(self.sim.data.ctrl))
    self.cmd_buffer.append(np.copy(np.array(action, dtype=np.float32)))
    for jnt_name in self.monitored_joints:
      self.joint_qpos_buffer[jnt_name].append(np.copy(self.sim.data.get_joint_qpos(jnt_name)))
      self.joint_qvel_buffer[jnt_name].append(np.copy(self.sim.data.get_joint_qvel(jnt_name)))
    for mcp_name in self.monitored_mocaps:
      qpos = np.zeros((7, ))
      qpos[:3] = self.sim.data.get_mocap_pos(mcp_name)
      qpos[3:] = self.sim.data.get_mocap_quat(mcp_name)
      self.mocap_qpos_buffer[mcp_name].append(qpos)
    for jnt_name in self.monitored_objects:
      self.object_qpos_buffer[jnt_name].append(np.copy(self.sim.data.get_joint_qpos(jnt_name)))

  def save_replay_buffer_pkl(self, file_path):
    """Saves the current replay buffer to a pickle file."""
    replay_buffer = {
        # meta
        'monitored_joints' : self.monitored_joints,
        'actuated_joints' : self.actuated_joints,
        'monitored_mocaps' : self.monitored_mocaps,
        'monitored_objects' : self.monitored_objects,
        # data frames
        'step_buffer' : self.step_buffer,
        'time_elapsed' : self.time_elapsed,
        'rgb_buffer' : self.rgb_buffer,
        'cmd_buffer' : self.cmd_buffer,
        'ctrl_buffer' : self.ctrl_buffer,
        'joint_qpos_buffer' : self.joint_qpos_buffer,
        'joint_qvel_buffer' : self.joint_qvel_buffer,
        'mocap_qpos_buffer' : self.mocap_qpos_buffer,
        'object_qpos_buffer' : self.object_qpos_buffer,
    }
    with open(file_path, 'wb') as f:
      pickle.dump(replay_buffer, f)


  # --- tfrecord frame recorder

  def _record_data_frame(self, action):
    """
    Feeds a data frame to the trecorder.
    Called by PickAndPlaceEnv.step()
    """
    frame = {}
    frame['step'] = self.ts
    frame['ts'] = self.sim.data.time
    _h, _w = self.frame_res
    if self.rendering_mode == 'tfrecord':
      cam_name = 'external_camera_1'  # TODO: make parameter!
      rgb, depth = self.sim.render(width=_w, height=_h, camera_name=cam_name, depth=True)
      frame['rgb'] = np.copy(rgb[::-1, :, :])  # original image is upside-down, so flip it
      frame['depth'] = np.copy(depth[::-1])  # original image is upside-down, so flip it
      # # DEBUG: dump depth frame
      # img_path = os.path.join(self.work_dir, 'depth_%05d.png' % self.ts)
      # scipy.misc.imsave(img_path, depth[::-1])
    else:
      frame['rgb'] = np.zeros((_h, _w, 3))  # TODO: append frames only in offscreen mode!
      frame['depth'] = np.zeros((_h, _w))  # TODO: append frames only in offscreen mode!
    frame['cmd'] = np.copy(np.array(action, dtype=np.float32))
    frame['ctrl'] = np.copy(self.sim.data.ctrl)
    for jnt_name in self.monitored_joints:
      key_qpos = 'joint_qpos-%s' % (jnt_name, )
      key_qvel = 'joint_qvel-%s' % (jnt_name, )
      frame[key_qpos] = np.copy(self.sim.data.get_joint_qpos(jnt_name))
      frame[key_qvel] = np.copy(self.sim.data.get_joint_qvel(jnt_name))
    for mcp_name in self.monitored_mocaps:
      key_mcp = 'mocap_qpos-%s' % (mcp_name, )
      qpos = np.zeros((7, ))
      qpos[:3] = self.sim.data.get_mocap_pos(mcp_name)
      qpos[3:] = self.sim.data.get_mocap_quat(mcp_name)
      frame[key_mcp] = qpos
    for jnt_name in self.monitored_objects:
      key_obj_jnt = 'object_qpos-%s' % (jnt_name, )
      frame[key_obj_jnt] = np.copy(self.sim.data.get_joint_qpos(jnt_name))
    # record goal & object qpos
    goal_jnt_name = '%s:joint' % (self.task_goal, )
    obj_jnt_name = '%s:joint' % (self.task_object, )
    frame['goal_qpos'] = np.copy(self.sim.data.get_joint_qpos(goal_jnt_name))
    frame['obj_qpos'] = np.copy(self.sim.data.get_joint_qpos(obj_jnt_name))
    self.tfrecorder.feed(frame)

  def start_tfrecorder(self, record_name):
    """Starts a new tfrecorder for the given `record_name`."""
    sequence_context = dict(self.meta_info_dict)
    sequence_context['task_goal'] = self.task_goal
    sequence_context['task_object'] = self.task_object
    self.tfrecorder = self.tfrecorder_fn(sequence_context, record_name)

  def save_tfrecord(self):
    """Saves the current tfrecord to file."""
    if self.tfrecorder:
      self.tfrecorder.finalize(compression='zlib')
    else:
      err_msg = "Cannot save a tfrecord because no tfrecorder is running!"
      raise RuntimeError(err_msg)


  # --- state loading APIs

  def enqueue_reset_state(self, state):
    """
    Puts a reset state into the reset queue. Will be used upon next env.reset().
    :param state:
      init: {<jnt_name> : <qpos_0>}
      robot: {<mcp_name> : <qpos_0>}
      task:
      target: (optional)
    """
    self.reset_queue.append(state)


  # --- basic env APIs

  def step(self, action):
    """
    Invokes base step function but also increases timestep counter and
    appends record buffer.

    :param action:
      [dx, dy, dz, cmp_grp]
      dx, dy, dz: delta in EE position in cartesian world coordinates
      cmp_grp: -1 (close), 0 (no-op), 1 (open)
    """
    self._append_replay_buffer(action)
    if self.tfrecorder:
      self._record_data_frame(action)
    self.ts += 1
    return super().step(action)

  def _set_action(self, action):
    """
    Applies the action to the environment.

    :param action:
      [dx, dy, dz, cmp_grp]
      dx, dy, dz: delta in EE position in cartesian world coordinates
      cmp_grp: -1 (close), 0 (no-op), 1 (open)
    """
    # super()._set_action(action)  # old action semantics

    # clip cmp_grp to nearest int and map to ctrl (-0.005, 0, 0.05)
    ctrl_grp_dict = {
        -1 : -0.005,  # close gripper
        0 : 0.0,  # no-op
        1 : 0.05,  # open gripper
    }
    assert action.shape == (4,)
    action = action.copy()  # ensure that we don't change the action outside of this scope
    pos_ctrl, gripper_cmd = action[:3], action[3]
    cmd_grp = int(np.rint(gripper_cmd))
    gripper_ctrl = ctrl_grp_dict[cmd_grp]

    pos_ctrl *= 0.05  # limit maximum change in position
    rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion

    gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
    assert gripper_ctrl.shape == (2,)
    if self.block_gripper:
      gripper_ctrl = np.zeros_like(gripper_ctrl)
    action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

    # Apply action to simulation.
    robo_utils.ctrl_set_action(self.sim, action)
    robo_utils.mocap_set_action(self.sim, action)

  def _reset_sim(self):
    """
    Called by gym.envs.robotics.RobotEnv.reset().
    Resets the environment and randomizes the starting conditions.
    """
    self.sim.set_state(self.initial_state)

    color_map = {  # geom_name -> RGBA
        'object0' : np.array([1, 0, 0, 1]),  # red cube
        'object1' : np.array([1, 1, 0, 1]),  # yellow cube
        'goal0' : np.array([0, 0, 1, 1]),  # blue goal
        'goal1' : np.array([0, 1, 0, 1]),  # green goal
    }
    obj_names = [n for n in self.sim.model.site_names \
      if n.startswith('object') or n.startswith('goal')]
    goal_names = [n for n in obj_names if n.startswith('goal')]
    cube_names = [n for n in obj_names if n.startswith('object')]

    # re-color objects
    for obj_name in obj_names:
      if obj_name in color_map:
        color = color_map[obj_name]
      else:  # object not to be modified
        continue
      try:  # re-color single-geom body
        geom_id = self.sim.model.geom_name2id(obj_name)
        self.sim.model.geom_rgba[geom_id] = color
      except:  # re-color multi-geom body
        geom_names = [n for n in self.sim.model.geom_names if n.startswith(obj_name)]  # TODO: use regex here!
        for gn in geom_names:
          gid = self.sim.model.geom_name2id(gn)
          self.sim.model.geom_rgba[gid] = color

    # set starting positions
    if len(self.reset_queue) > 0:  # take reset state from queue
      reset_state = self.reset_queue.popleft()  # FIFO queue!
      print("Reset environment to queued state:")
      pprint.pprint(reset_state['init'])
      pprint.pprint(reset_state['robot'])
      pprint.pprint(reset_state['task'])
      init_state = reset_state['init']
      robot_state = reset_state['robot']
      task = reset_state['task']
      if 'target' in reset_state:
        self.target_frame = reset_state['target']
        if isinstance(self.target_frame, list):
          print("Registered a target frame of shape %s" % (self.target_frame[0].shape, ))
        else:
          print("Registered a target frame of shape %s" % (self.target_frame.shape, ))
      for jnt_name, qpos in init_state.items():
        # qpos[2] = self.height_offset
        qpos[2] += 0.025  # TODO: get table & object height dynamically!
        self.sim.data.set_joint_qpos(jnt_name, qpos)
      for mcp_name, qpos in robot_state.items():
        self.sim.data.set_mocap_pos(mcp_name, qpos[:3])
        self.sim.data.set_mocap_quat(mcp_name, qpos[3:])
      self.task_goal = task['goal']
      self.task_object = task['object']
    else:  # randomize init state & task
      # TODO: compute workspace extensions dynamically!
      shapes2dims = {
        'push-pad1-cube1' : {'minmax_x' : (1.2, 1.3), 'minmax_y' : (0.450, 1.050), 'tiling_xy' : (6, 8), 'goal_offset_x' : 0.1},
        'push-pad1-cube2' : {'minmax_x' : (1.175, 1.4), 'minmax_y' : (0.5, 1), 'tiling_xy' : (2, 3), 'goal_offset_x' : 0.125},
        'push-pad2-cube1' : {'minmax_x' : (1.175, 1.4), 'minmax_y' : (0.5, 1), 'tiling_xy' : (2, 3), 'goal_offset_x' : 0.125},
        'push-pad2-cube2' : {'minmax_x' : (1.175, 1.4), 'minmax_y' : (0.5, 1), 'tiling_xy' : (2, 3), 'goal_offset_x' : 0.125},
      }
      dims = shapes2dims[self.shapes]
      spawn_points = grids.sample_spawn_grid(len(obj_names), dims['minmax_x'], dims['minmax_y'], dims['tiling_xy'])
      for obj_idx, (obj_name, spawn_pos) in enumerate(zip(obj_names, spawn_points)):
        # teleport object to new position
        object_xpos = np.array(spawn_pos)
        # set the goal further along the x axis
        if obj_name in goal_names:
          object_xpos[0] += dims['goal_offset_x']
        # rot_z = np.random.rand() * 2 * np.pi
        rot_z = 0
        object_quat = rotations.euler2quat(np.array([0, 0, rot_z]))
        jnt_name = '%s:joint' % (obj_name, )
        object_qpos = self.sim.data.get_joint_qpos(jnt_name)
        assert object_qpos.shape == (7,)
        object_qpos[:2] = object_xpos
        # object_qpos[2] = self.height_offset
        object_qpos[2] = 0.27 + 0.025  # TODO: get table & object height dynamically!
        object_qpos[3:] = object_quat
        self.sim.data.set_joint_qpos(jnt_name, object_qpos)
      # randomize intial robot state
      # TODO: remove hard-coded constants!
      robot_xpos0 = np.array([1.3419, 0.7491, 0.8]) \
        + np.array(sample_point_within_sphere(radius=0.03))  # 1.3419, 0.7491, 0.555
      self.sim.data.set_mocap_pos('robot0:mocap', robot_xpos0)
      self.sim.data.set_mocap_quat('robot0:mocap', np.array([1., 0., 1., 0.]))
      # randomize task
      self.task_goal = np.random.choice(goal_names)
      self.task_object = np.random.choice(cube_names)

    # apply changes to sim
    # self.sim.forward()
    for _ in range(10):  # step forward to drag mocap-controlled EE into place
      self.sim.step()
    self.ts = 0
    self._reset_replay_buffer()

    return True

  def _env_setup(self, initial_qpos):
    """
    Called by gym.envs.robotics.RobotEnv().
    Defines the initial environment configuration.
    """
    for name, value in initial_qpos.items():
      self.sim.data.set_joint_qpos(name, value)
    robo_utils.reset_mocap_welds(self.sim)
    self.sim.forward()

    # Move end effector into position.
    gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
    print(">>> Gripper XPOS_0:", gripper_target)
    gripper_rotation = np.array([1., 0., 1., 0.])
    self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
    self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
    for _ in range(10):
      self.sim.step()

    # Extract information for sampling goals.
    self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
    # if self.has_object:
    #   self.height_offset = self.sim.data.get_site_xpos('object0')[2]


  # --- rendering APIs

  def _init_rendering(self):
    if self.rendering_mode != 'viewer':
      # switch off debugging visuals
      site_names = [n for n in self.sim.model.site_names \
        if n.startswith('object') or n.startswith('goal')]
      site_names.extend(list(DEBUG_SITE_VISUALS.keys()))
      geom_names = list(DEBUG_GEOM_VISUALS.keys())
      for site_name in site_names:
        site_id = self.sim.model.site_name2id(site_name)
        self.sim.model.site_rgba[site_id][-1] = 0.0  # RGBA alpha=0 -> transparent
      for geom_name in geom_names:
        geom_id = self.sim.model.geom_name2id(geom_name)
        self.sim.model.geom_rgba[geom_id][-1] = 0.0  # RGBA alpha=0 -> transparent
      self.sim.forward()  # transfer model changes into sim

  # not modified
  def _render_callback(self):
    """
    Called by gym.envs.robotics.RobotEnv.render() before _get_viewer().render().
    """
    # visualize target
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')
    self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    self.sim.forward()

  def render_extended(self):
    """
    Calls the internal robot_env.render() function.
    Collects the results and handles video / tfrecord saving.
    """
    if self.rendering_mode == 'viewer':
      self.render(mode='human')
    elif self.rendering_mode == 'video':
      # rgb_frame = self.render(mode='rgb_array')
      _h, _w = self.frame_res
      _cam_name = 'external_camera_1'
      rgb_frame = self.sim.render(width=_w, height=_h, camera_name=_cam_name)
      rgb_frame = rgb_frame[::-1, :, :]  # original image is upside-down, flip
      self.recorder.feed(rgb_frame)
    elif self.rendering_mode == 'tfrecord':
      pass  # rendering done in _record_data_frame()


  # --- TODO: RL APIs

  # TODO: to be modified!
  def sample_goal(self, goal=None):
    """Resample and set goal."""
    if goal is None:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
      goal[2] += 0.05
    self.goal = goal.copy()
    return goal.copy()

  # TODO: to be modified!
  def _sample_goal(self):
    """
    Called by gym.envs.robotics.RobotEnv __init__(), reset().
    Samples a new goal for the top level gym.GoalEnv.
    Goal is target position of object for pushing.
    """
    if self.has_object:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      goal += self.target_offset
      # goal[2] = self.height_offset
      goal[2] = 0.27 + 0.025  # TODO: get table & object height dynamically!
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return goal.copy()

  # TODO: to be modified!
  def _get_obs(self):
    """
    Called by gym.envs.robotics.RobotEnv: __init__(), step(), reset()
    Returns the current observation as dict in the gym.GoalEnv format.
    """
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = robo_utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.squeeze(object_pos.copy())
    obs = np.concatenate([
        grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
        object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    ])

    return {
        'observation': obs.copy(),
        'achieved_goal': achieved_goal.copy(),
        'desired_goal': self.goal.copy(),
    }
