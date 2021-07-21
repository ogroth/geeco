"""
Run script for a pick and place Gym environment featuring a Fetch robot.
Can be used to collect expert trajectories, replay recorded trajectories or
run and evaluate a controller.

Example calls:

* Simulation of expert trajectories with onscreen rendering:
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so" python3 gym_pickplace.py \
  --rendering_mode viewer

* Simulation of expert trajectories with video rendering:
python3 gym_pickplace.py \
  --rendering_mode video

* Running a controller:
python3 gym_pickplace.py \
  --shapes pad1-cube1 \
  --sim_mode controller \
  --model_dir ../models/geeco-f/pick11 \
  --goal_condition none \
  --rendering_mode video \
  --debug
"""

import argparse
import os
import pickle
import csv
import json

import gym
from gym.envs.registration import register
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.geeco_gym import load_target_frame
from mj_engine.engine.recorder import MjVideoRecorder
from models.e2evmc.predictor import E2EVMCPredictor, GoalE2EVMCPredictor
from utils.runscript import save_run_command


#region: CLI arguments

ARGPARSER = argparse.ArgumentParser(
    description='Collect data for a simple block stacking task with a Fetch robot in a\
      gym environment.')
ARGPARSER.add_argument(
    '--wrk_dir', type=str, default='../logs/gym_pickplace',
    help="The path to the working directory for this run (where logs, videos \
      etc. will be stored).")
ARGPARSER.add_argument(
    '--shapes', type=str, default='pad2-cube2',
    help="The shape set used for stacking. \
      Available: pad1-cube1 | pad2-cube1 | pad1-cube2 | pad2-cube2 | \
      pad2-cube2-clutter4 | pad2-cube2-clutter12 | \
      ball-cup | bridge-pad | diamond-pad | nut-cone")
ARGPARSER.add_argument(
    '--sim_mode', type=str, default='collect',
    help="The simulation mode to use: \
      collect = collect expert data \
      replay = replay from control buffer \
      random = random freestyle control \
      controller = run with controller")
ARGPARSER.add_argument(
    '--max_episode_steps', type=int, default=-1,
    help="Max. simulation steps in one episode. If < 0, use defaults.")
ARGPARSER.add_argument(
    '--dry_run', default=False, action='store_true',
    help="Saves images of initial configurations without running the simulation. \
      Only effective in 'collect' mode.")
ARGPARSER.add_argument(
    '--init_states', type=str, default='',
    help="The path to a CSV file containing initial qpos of objects.")
# --- data collection options
ARGPARSER.add_argument(
    '--start_idx', type=int, default=0,
    help="Index of the first record to be collected.")
ARGPARSER.add_argument(
    '--end_idx', type=int, default=100,
    help="Index of the last record to be collected.")
# --- replay options
ARGPARSER.add_argument(
    '--replay_buffer', type=str, default='../data/gym-pick-pad2-cube2-v4/data/replay_buffer_0001.pkl',
    help="The path to the buffer file to be replayed. Only used with --sim_mode=replay.")
# --- control options
ARGPARSER.add_argument(  # determines which controller to use
    '--controller', type=str, default='e2evmc',
    help="Controller model to use. Options are: e2evmc. \
      e2evmc = E2E visuomotor control (reflex), works in all goal_conditon modes")
ARGPARSER.add_argument(  # determines which predictor mode to use
    '--goal_condition', type=str, default='none',
    help="Conditioning mode of the reflex. Options are: none | target | inter_targets. \
      none = no goal provided, unconditional reflex \
      target = target image provided, conditional reflex \
      inter_targets = intermediate target images provided, conditional reflex")
ARGPARSER.add_argument(
    '--model_dir', type=str, default='../models/geeco-f/pick11',
    help="Directory from where the controller model is loaded.")
ARGPARSER.add_argument(
    '--checkpoint_name', type=str, default=None,
    help="Specific checkpoint to load. If none, load latest in model_dir.")
ARGPARSER.add_argument(
    '--dataset_dir', type=str, default='../data/gym-pick-pad2-cube2-v4/',
    help="Dataset from which evaluation data is loaded.")
ARGPARSER.add_argument(
    '--tfrecord_list', type=str, default='',
    help="The path to txt file containing tfrecords to be evaluated on.")
# --- randomization options
ARGPARSER.add_argument(
    '--background_video', type=str, default='',
    help="The path to the distractor video to be looped in the background. If empty, no video is used.")
# --- rendering options
ARGPARSER.add_argument(
    '--rendering_mode', type=str, default='viewer',
    help="The rendering mode to use: viewer | video | tfrecord")
ARGPARSER.add_argument(
    '--frame_res', type=int, nargs=2, default=[256, 256],
    help="Resolution of the recorded camera frames as (width, height).")
ARGPARSER.add_argument(
    '--observation_format', type=str, default='rgb',
    help='Observation data to be used (sets img_channels): rgb | rgbd. \
      Only effective in `controller` mode.')
# --- debug options
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debug output.")

#endregion


#region: constants

NAME_GRIPPER = 'robot0:grip'
NAME_TABLE = 'table0'
OFFSET_HEIGHT_PRE_GRASP = 0.05  # gripper offset above target object
DIST_PRE_GRASP = 0.005  # distance to target object for pre-grasp pose
DIST_GRASP = 0.002  # distance to target object for grasp pose (defines firmness of grip)
DIST_ON_TOP = 0.175  # distance for placement position on top of a target
DIST_GOAL = 0.01  # radius around goal pos, positions within are considered 'in goal'
SIZE_BOX = 0.05  # side-length of a box
MULT_POS_ACTION = 6.0  # multiplier for pos action (for faster movement to target pos)
TOL_GRIPPER_RELEASE = 0.0001  # tolerance between gripper state and full open (for release)
PAUSE_AFTER_DROP = 10  # number of simulation steps for the robot to sit idle after placing an object (to let the stack settle)
CMD_GRIPPER_OPEN = 1.0
CMD_GRIPPER_CLOSE = -1.0
CMD_GRIPPER_NOOP = 0.0

GOAL_NAMES = {  # env_name -> [goal_name]
    'pad1-cube1' : ['goal0'],
    'pad2-cube1' : ['goal0', 'goal1'],
    'pad1-cube2' : ['goal0'],
    'pad2-cube2' : ['goal0', 'goal1'],
    'pad2-cube2-clutter4' : ['goal0', 'goal1'],
    'pad2-cube2-clutter12' : ['goal0', 'goal1'],
    # generalization scenarios
    'ball-cup' : ['goal0', 'goal1'],
    'bridge-pad' : ['goal0', 'goal1'],
    'diamond-pad' : ['goal0', 'goal1'],
    'nut-cone' : ['goal0', 'goal1'],
}
CUBE_NAMES = {  # env_name -> [cube_name]
    'pad1-cube1' : ['object0'],
    'pad2-cube1' : ['object0'],
    'pad1-cube2' : ['object0', 'object1'],
    'pad2-cube2' : ['object0', 'object1'],
    'pad2-cube2-clutter4' : ['object0', 'object1'],
    'pad2-cube2-clutter12' : ['object0', 'object1'],
    # generalization scenarios
    'ball-cup' : ['goal0', 'goal1'],
    'bridge-pad' : ['goal0', 'goal1'],
    'diamond-pad' : ['goal0', 'goal1'],
    'nut-cone' : ['goal0', 'goal1'],
}

#endregion


#region: I/O helper

def _load_reset_queue_v2(env, reset_states_path, start_idx, end_idx):
  with open(reset_states_path) as fp:
    reader = csv.reader(fp, delimiter=';')
    iterator = iter(reader)
    # parse CSV header: extract joint names
    header_row = next(iterator)
    state_header = header_row[:-2]  # last two fields are task_goal & task_object
    task_header = header_row[-2:]
    num_joints = len(state_header) // 7
    joint_names = [state_header[i * 7].split('::')[0] for i in range(num_joints)]
    for i in range(0, end_idx):
      try:
        row = next(iterator)
      except StopIteration:
        break
      if i < start_idx:
        continue
      # each parsed row is one init state and task definition
      state_row = row[:-2]
      task_row = row[-2:]
      state_row = [float(e) for e in state_row]
      qpos_list = np.split(np.array(state_row), num_joints)
      state_dict = dict(list(zip(joint_names[:-1], qpos_list[:-1])))
      robot_dict = dict([(joint_names[-1], qpos_list[-1])])
      task_dict = {
          'goal' : task_row[0].split(','),
          'object' : task_row[1].split(','),
      }
      reset_state = {
          'init' : state_dict,
          'task' : task_dict,
          'robot' : robot_dict,
      }
      env.enqueue_reset_state(reset_state)

def _load_reset_queue_v3(env, reset_states_path, tfrecord_list_path, dataset_dir, start_idx, end_idx):
  """Row in reset state CSV and lines in tfrecord list file must be aligned!"""
  with open(reset_states_path) as fp:
    reader = csv.reader(fp, delimiter=';')
    state_iterator = iter(reader)
    # parse CSV header: extract joint names
    header_row = next(state_iterator)
    state_header = header_row[:-2]  # last two fields are task_goal & task_object
    task_header = header_row[-2:]
    num_joints = len(state_header) // 7
    joint_names = [state_header[i * 7].split('::')[0] for i in range(num_joints)]
    # load tfrecord list
    with open(tfrecord_list_path) as fp2:
      tfrecord_list = fp2.read().split('\n')[:-1]
    record_iterator = iter(tfrecord_list)
    # only load tfrecords that are needed
    for i in range(0, end_idx):
      try:
        row = next(state_iterator)
        record_name = next(record_iterator)
      except StopIteration:
        break
      if i < start_idx:
        continue
      # each parsed row is one init state and task definition
      state_row = row[:-2]
      task_row = row[-2:]
      state_row = [float(e) for e in state_row]
      qpos_list = np.split(np.array(state_row), num_joints)
      state_dict = dict(list(zip(joint_names[:-1], qpos_list[:-1])))
      robot_dict = dict([(joint_names[-1], qpos_list[-1])])
      task_dict = {
          'goal' : task_row[0].split(','),
          'object' : task_row[1].split(','),
      }
      reset_state = {
          'init' : state_dict,
          'task' : task_dict,
          'target' : [load_target_frame(dataset_dir, record_name, load_depth=False)],  # HOTFIX: disabled depth!
          'robot' : robot_dict,
      }
      env.enqueue_reset_state(reset_state)

def _animate_trajectories(traj_ee, traj_obj, pred_traj_ee, pred_traj_obj):
  """
    traj_ee: gray
    traj_obj: red
    pred_traj_ee: blue
    pred_traj_obj: yellow
  """
  # reshape trajectory tensors
  traj_ee = np.stack(traj_ee)  # [t, xyz]
  traj_obj = np.stack(traj_obj)  # [t, xyz]
  pred_traj_ee = np.stack(pred_traj_ee)  # [t, xyz]
  pred_traj_obj = np.stack(pred_traj_obj)  # [t, xyz]
  # determine sequence length
  seq_length = np.min([
      traj_ee.shape[0], traj_obj.shape[0], pred_traj_ee.shape[0], pred_traj_obj.shape[0]
  ])
  # create figure and set limits
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  x_data = np.concatenate([
      traj_ee[:, 0], traj_obj[:, 0], pred_traj_ee[:, 0], pred_traj_obj[:, 0]
  ])
  y_data = np.concatenate([
      traj_ee[:, 1], traj_obj[:, 1], pred_traj_ee[:, 1], pred_traj_obj[:, 1]
  ])
  z_data = np.concatenate([
      traj_ee[:, 2], traj_obj[:, 2], pred_traj_ee[:, 2], pred_traj_obj[:, 2]
  ])
  ax.set_xlim(left=np.min(x_data), right=np.max(x_data))
  ax.set_ylim(bottom=np.min(y_data), top=np.max(y_data))
  ax.set_zlim(bottom=np.min(z_data), top=np.max(z_data))
  # internal anim func
  def _init():
    _traj_ee, = ax.plot3D([traj_ee[0, 0]], [traj_ee[0, 1]], [traj_ee[0, 2]], 'black')
    _traj_obj, = ax.plot3D([traj_obj[0, 0]], [traj_obj[0, 1]], [traj_obj[0, 2]], 'green')
    _pred_traj_ee, = ax.plot3D([pred_traj_ee[0, 0]], [pred_traj_ee[0, 1]], [pred_traj_ee[0, 2]], 'blue')
    _pred_traj_obj, = ax.plot3D([pred_traj_obj[0, 0]], [pred_traj_obj[0, 1]], [pred_traj_obj[0, 2]], 'red')
    return _traj_ee, _traj_obj, _pred_traj_ee, _pred_traj_obj
  def _animate(i):
    _traj_ee, = ax.plot3D(traj_ee[0:i, 0], traj_ee[0:i, 1], traj_ee[0:i, 2], 'black')
    _traj_obj, = ax.plot3D(traj_obj[0:i, 0], traj_obj[0:i, 1], traj_obj[0:i, 2], 'green')
    _pred_traj_ee, = ax.plot3D(pred_traj_ee[0:i, 0], pred_traj_ee[0:i, 1], pred_traj_ee[0:i, 2], 'blue')
    _pred_traj_obj, = ax.plot3D(pred_traj_obj[0:i, 0], pred_traj_obj[0:i, 1], pred_traj_obj[0:i, 2], 'red')
    return _traj_ee, _traj_obj, _pred_traj_ee, _pred_traj_obj
  # build animation
  ani = animation.FuncAnimation(  # interval: 40 ms = 25 Hz
      fig, _animate, init_func=_init, interval=40, blit=True, save_count=seq_length)
  return ani

def _animate_commands(cmd_ee, cmd_grp):
  """stub"""
  # reshape trajectory tensors
  cmd_ee = np.stack(cmd_ee)  # [t, Dxyz]
  cmd_grp = np.array(cmd_grp)  # [t, 1]
  # determine sequence length
  seq_length = np.min([cmd_ee.shape[0], cmd_grp.shape[0]])
  # create figure and set limits
  fig, (ax_cmd_dx, ax_cmd_dy, ax_cmd_dz, ax_cmd_grp) = plt.subplots(nrows=4, ncols=1, sharex=True)
  cmd_ee_range = np.max(cmd_ee) - np.min(cmd_ee)
  for ax in [ax_cmd_dx, ax_cmd_dy, ax_cmd_dz]:
    ax.set_xlim(left=0, right=seq_length)
    # ax.set_ylim(bottom=np.min(cmd_ee)-0.1*cmd_ee_range, top=np.max(cmd_ee)+0.1*cmd_ee_range)
    ax.set_ylim(bottom=-2.0, top=2.0)
  ax_cmd_grp.set_xlim(left=0, right=seq_length)
  ax_cmd_grp.set_ylim(bottom=-1.5, top=1.5)
  # internal anim func
  def _init():
    _cmd_dx, = ax_cmd_dx.plot([0], [cmd_ee[0, 0]], color='orange')
    _cmd_dy, = ax_cmd_dy.plot([0], [cmd_ee[0, 1]], color='orange')
    _cmd_dz, = ax_cmd_dz.plot([0], [cmd_ee[0, 2]], color='orange')
    _cmd_grp, = ax_cmd_grp.plot([0], [cmd_grp[0, 0]], color='orange')
    return _cmd_dx, _cmd_dy, _cmd_dz, _cmd_grp
  def _animate(i):
    _cmd_dx, = ax_cmd_dx.plot(np.arange(0, i), cmd_ee[0:i, 0], color='orange')
    _cmd_dy, = ax_cmd_dy.plot(np.arange(0, i), cmd_ee[0:i, 1], color='orange')
    _cmd_dz, = ax_cmd_dz.plot(np.arange(0, i), cmd_ee[0:i, 2], color='orange')
    _cmd_grp, = ax_cmd_grp.plot(np.arange(0, i), cmd_grp[0:i, 0], color='orange')
    return _cmd_dx, _cmd_dy, _cmd_dz, _cmd_grp
  # build animation
  ani = animation.FuncAnimation(  # interval: 40 ms = 25 Hz
      fig, _animate, init_func=_init, interval=40, blit=True, save_count=seq_length)
  return ani

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

#endregion


#region: sub-routines for expert trajectory collection

def _move_to_pre_grasp(env, obj_name: str):
  """Move gripper into pre-grasp pose for object."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # first make the gripper go slightly above the object
  object_oriented_goal = object_rel_pos.copy()
  object_oriented_goal[2] += OFFSET_HEIGHT_PRE_GRASP
  # set goal
  env.sample_goal(object_oriented_goal)
  # movement loop
  while np.linalg.norm(object_oriented_goal) >= DIST_PRE_GRASP \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[2] += OFFSET_HEIGHT_PRE_GRASP
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(object_oriented_goal)):
      action[i] = object_oriented_goal[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_OPEN
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # pre-grasp pose reached

def _grasp(env, obj_name: str):
  """Grasp object."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # movement loop
  while np.linalg.norm(object_rel_pos) >= DIST_GRASP \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(object_rel_pos)):
      action[i] = object_rel_pos[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # object has been grasped

def _move_to_post_grasp(env, obj_name: str):
  """Move gripper into post-grasp pose for object."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  goal_grip_pos = grip_pos.copy()
  goal_grip_pos[2] += OFFSET_HEIGHT_PRE_GRASP
  # set goal
  env.sample_goal(goal_grip_pos)
  # first make the gripper go slightly above the object
  diff_grip_pos = goal_grip_pos - grip_pos
  # movement loop
  while np.linalg.norm(diff_grip_pos) >= DIST_PRE_GRASP \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(diff_grip_pos)):
      action[i] = diff_grip_pos[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    diff_grip_pos = goal_grip_pos - grip_pos
  return True  # post-grasp pose reached

def _move(env, obj_name: str, goal_pos):
  """Move object to a goal position. Object needs to be grasped first!"""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # set goal
  env.sample_goal(goal_pos)
  # movement loop
  while np.linalg.norm(goal_pos - object_pos) >= DIST_GOAL \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(goal_pos - object_pos)):
      action[i] = (goal_pos - object_pos)[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # object has been moved to goal position

def _drop(env, obj_name: str):
  """Open gripper and drop object."""
  gripper_open = np.array([CMD_GRIPPER_OPEN, CMD_GRIPPER_OPEN])
  gripper_act = np.array([  # only gripper in ctrl array, remaining arm pose is set via mocap array
      env.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint'),
      env.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint'),
  ])
  # movement loop
  while np.linalg.norm(gripper_act - gripper_open) >= TOL_GRIPPER_RELEASE \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    # adjust action
    # xpos_grp = env.sim.data.get_site_xpos('robot0:grip')
    action = [0, 0, OFFSET_HEIGHT_PRE_GRASP / 2, CMD_GRIPPER_OPEN]
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    gripper_act = np.array([
        env.sim.data.get_joint_qpos('robot0:l_gripper_finger_joint'),
        env.sim.data.get_joint_qpos('robot0:r_gripper_finger_joint'),
    ])
  return True  # gripper is fully open and has released object

def _idle(env, idle_steps):
  """Lets the robot sit idle for a number of steps."""
  step_cnt = 0
  while step_cnt < idle_steps \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    action = [0, 0, 0, 0]
    obs, reward, done, info = env.step(action)
    step_cnt += 1
  return

def _on_top(env, obj_top: str, obj_bottom: str):
  """Computes a goal position for `obj_top` on top of `obj_bottom`."""
  xpos_bottom = env.sim.data.get_site_xpos(obj_bottom)
  xpos_top = xpos_bottom.copy()
  xpos_top[2] += DIST_ON_TOP
  return xpos_top

def _get_obj_heights(env):
  """Returns the names and heights (site_xpos) of all objects, sorted desc."""
  obj_names = [n for n in env.sim.model.site_names \
    if n.startswith('object') or n.startswith('goal')]  # TODO: use regex here!
  obj_heights = [env.sim.data.get_site_xpos(obj_name)[2] for obj_name in obj_names]
  result = list(zip(obj_names, obj_heights))
  result.sort(key=lambda t: t[1])
  result.reverse()
  return result

# NOTE: hard-coded for box towers!
def _get_stack_height(env):
  """Returns the height of the highest stack."""
  offset_table = env.sim.data.get_body_xpos(NAME_TABLE)[2] * 2  # body center is at half-height
  _obj, _h = _get_obj_heights(env)[0]  # get highest object site xpos
  # TODO: make box half-heights constant
  max_height = _h + SIZE_BOX / 2
  print(max_height, offset_table)
  num_stacked = (max_height - offset_table) / SIZE_BOX
  return num_stacked

def _stack(env, obj_names, goal_name):
  num_waypoints = 1  # TODO: compute more waypoints for collision-free trajectory
  obj_moved = []  # keep track of the objects which have already been moved
  # set goals
  obj_moved.append(goal_name)
  for obj_idx, obj_name in enumerate(obj_names):
    _move_to_pre_grasp(env, obj_name)
    _grasp(env, obj_name)
    _move_to_post_grasp(env, obj_name)
    for i in range(num_waypoints):
      # move object on top of currently highest one
      obj_by_height = _get_obj_heights(env)
      for _obj, _h in obj_by_height:
        if _obj in obj_moved:  # ensure that bottom object has already been moved!
          obj_bottom, cur_h = _obj, _h
          break
      goal = _on_top(env, obj_name, obj_bottom)
      _move(env, obj_name, goal)
    _drop(env, obj_name)
    obj_moved.append(obj_name)
    _idle(env, PAUSE_AFTER_DROP)
    stack_height = int(np.rint(_get_stack_height(env)))
    print(">> Current stack height", stack_height)
  while env.ts < env._max_episode_steps:
    env.render_extended()
    action = [0, 0, 0, 0]
    obs, reward, done, info = env.step(action)
  stack_height = int(np.rint(_get_stack_height(env)))  # TODO: save
  print(">> Final stack height", stack_height)

#endregion


#region: sub-routines for controller evaluation

# TODO: refactor into application constants / make parameter
OBJ_VICINITY = 0.0625  # radius around manipulated object; determines vicinity test
GRASP_VICINITY = 0.025  # distance between gripper and manipulated object; determines grasp test
GOAL_VICINITY = 0.05  # radius around goal object; determines task success test

def _eval_object_vicinity(env, obj_name: str):
  """Checks whether the gripper has come close enough to the object."""
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  dist = np.linalg.norm(object_pos - grip_pos)
  return dist <= OBJ_VICINITY

def _eval_grasp_success(env, obj_name: str):
  """Checks whether the gripper has grasped the object."""
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  dist = np.linalg.norm(object_pos - grip_pos)
  return dist <= GRASP_VICINITY

def _eval_task_success(env, obj_name: str, goal_name: str):
  """Checks whether the manipulated object is near the goal."""
  object_pos = env.sim.data.get_site_xpos(obj_name)
  goal_pos = env.sim.data.get_site_xpos(goal_name)
  dist = np.linalg.norm(goal_pos - object_pos)
  return dist <= GOAL_VICINITY

def _eval_goal_dist(env, obj_name: str, goal_name: str):
  """Returns the distance of the manipulated object to the goal."""
  object_pos = env.sim.data.get_site_xpos(obj_name)
  goal_pos = env.sim.data.get_site_xpos(goal_name)
  dist = np.linalg.norm(goal_pos - object_pos)
  return dist

# endregion


#region: main

def main(args):
  # --- set up directories
  wrk_dir = os.path.join(args.wrk_dir, args.sim_mode)
  os.makedirs(wrk_dir, exist_ok=True)
  run_cmd_path = save_run_command(argparser=ARGPARSER, run_dir=wrk_dir)

  # --- environment registration
  for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'work_dir' : wrk_dir,
        'shapes' : args.shapes,
        'reward_type' : reward_type,
        'rendering_mode' : args.rendering_mode,
        'frame_res' : tuple(args.frame_res),
    }
    if args.sim_mode == 'controller':  # max_episode_steps depend
      if args.max_episode_steps < 0:  # use defaults
        max_episode_steps = 200  # normal controller execution (2x expert demonstration time)
      else:
        max_episode_steps = args.max_episode_steps
    else:
      max_episode_steps = 100  # time for expert demonstration
    register(
        id='PickAndPlaceEnv{}-v1'.format(suffix),
        entry_point='geeco_gym:PickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
    )

  # --- build gym environment
  env = gym.make('PickAndPlaceEnv-v1')
  env.reset()
  print("Initial environment reset done.")

  # --- constants
  goal_names = GOAL_NAMES[args.shapes]
  cube_names = CUBE_NAMES[args.shapes]

  # --- queue up intial environment configurations
  if os.path.isfile(args.init_states):
    if args.sim_mode == 'collect':
      _load_reset_queue_v2(env, args.init_states, args.start_idx, args.end_idx)
    elif args.sim_mode == 'controller':
      if args.goal_condition == 'none':
        _load_reset_queue_v2(env, args.init_states, args.start_idx, args.end_idx)
      elif args.goal_condition == 'target':
        _load_reset_queue_v3(
            env, args.init_states, args.tfrecord_list, args.dataset_dir, args.start_idx, args.end_idx)
  else:
    err_msg = ">>> Couldn't load initial states from %s! Defaulting to random initialization." \
      % (args.init_states, )
    print(err_msg)

  # --- set up domain randomization
  randomize_background = False
  if args.background_video != '':
    randomize_background = True
    bg_video_cycler = VideoCycler(args.background_video)
    from mujoco_py.modder import TextureModder
    texmodder = TextureModder(env.sim)

  # --- set up predictor from model_dir
  if args.sim_mode == 'controller':
    # E2EVMC
    if args.controller == 'e2evmc':
      if args.goal_condition == 'none':
        predictor = E2EVMCPredictor(args.model_dir, args.checkpoint_name)
      elif args.goal_condition == 'target':
        predictor = GoalE2EVMCPredictor(args.model_dir, args.checkpoint_name)
      else:
        err_msg = "Unknown goal condition: %s!" % (args.goal_condition, )
        raise ValueError(err_msg)
    else:
      err_msg = "Unknown controller model '%s'" % (args.controller, )
      raise ValueError(err_msg)
    # --- register video recorders
    recorders = {}
    recorders['rgb'] = MjVideoRecorder(  # TODO: set parameters via config
        ctx_name='default', ctx_type='rgb', cam_name='cam_default',
        record_name='observation', record_dir=wrk_dir,
        res_height=256, res_width=256)
    if args.debug:
      if predictor.cfg.proc_obs == 'dynimg':  # register recorder for dynamic buffer
        recorders['dynbuff'] = MjVideoRecorder(  # TODO: set parameters via config
            ctx_name='default', ctx_type='rgb', cam_name='cam_default',
            record_name='dynbuff', record_dir=wrk_dir,
            res_height=256, res_width=256)
      if predictor.cfg.proc_tgt == 'dyndiff':  # register recorder for dynamic buffer
        recorders['dyndiff'] = MjVideoRecorder(  # TODO: set parameters via config
            ctx_name='default', ctx_type='rgb', cam_name='cam_default',
            record_name='dyndiff', record_dir=wrk_dir,
            res_height=256, res_width=256)

  # --- set up evaluation data structures
  if args.sim_mode == 'controller':
    eval_results = []
    episode_eval_spec = [
        ('episode_id', 0),
        ('obj_vicinity', 0),  # binary success indicator
        ('grasp_success', 0),  # binary success indicator
        ('task_success', 0),  # binary success indicator
        ('init_goal_dist', 0),  # set at start of episode
        ('min_goal_dist', 1000),  # updated via min operator
        ('max_goal_dist', 0),  # updated via max operator
        ('final_goal_dist', 0),  # set at end of episode
        ('video_file', ''),  # path to video file
    ]
    eval_spec_fields = [t[0] for t in episode_eval_spec]
    report_path = os.path.join(wrk_dir, 'eval_results.csv')
    csv_report = open(report_path, 'w', newline='')
    writer = csv.DictWriter(csv_report, fieldnames=eval_spec_fields, delimiter=';')
    writer.writeheader()

  # --- main loop
  for i in tqdm(range(args.start_idx, args.end_idx)):
    obs = env.reset()
    episode_id = i + 1
    print("ITERATION NUMBER %d / %d" % (episode_id, args.end_idx))

    # --- collect data and save replay buffer (and video / tfrecord)
    if args.sim_mode == 'collect':
      if args.dry_run:  # only save initial image
        _h, _w = args.frame_res
        _cam_name = 'external_camera_1'
        rgb_frame = env.sim.render(width=_w, height=_h, camera_name=_cam_name)
        rgb_frame = rgb_frame[::-1, :, :]  # original image is upside-down, flip
        rgb_frame = rgb_frame / 255.0  # normalize RGB for predictor feeding
        img_path = os.path.join(wrk_dir, 'init_%04d.png' % (episode_id, ))
        # scipy.misc.imsave(img_path, rgb_frame)
        Image.fromarray((rgb_frame * 255).astype(np.uint8)).save(img_path)
        continue
      # get task information from env
      obj_names = env.task_object
      goal_name = env.task_goal[0]
      # save meta information
      meta_info_dict = env.encoding_meta._asdict()
      meta_info_path = os.path.join(wrk_dir, 'meta_info.json')
      with open(meta_info_path, 'w') as fp:
        json.dump(meta_info_dict, fp, indent=2, sort_keys=True)
      # start tfrecorder
      if args.rendering_mode == 'tfrecord':
        record_name = 'replay_buffer_%04d' % (episode_id, )
        env.start_tfrecorder(record_name)
      # perform stacking
      _stack(env, obj_names=obj_names, goal_name=goal_name)
      # save as pkl
      save_path = os.path.join(wrk_dir, 'replay_buffer_%04d.pkl' % (episode_id, ))
      env.save_replay_buffer_pkl(save_path)
      # save as tfrecord
      if args.rendering_mode == 'tfrecord':
        env.save_tfrecord()
      # save as video
      if env.rendering_mode == 'video':
        env.recorder.flush()

    # --- replay a buffer
    elif args.sim_mode == 'replay':
      with open(args.replay_buffer, 'rb') as f:
        replay_buffer = pickle.load(f)
      # read meta
      operated_joints = replay_buffer['monitored_joints']
      operated_mocaps = replay_buffer['monitored_mocaps']
      operated_actuators = replay_buffer['actuated_joints']
      operated_objects = replay_buffer['monitored_objects']
      # read buffers
      joint_qpos_buffer = replay_buffer['joint_qpos_buffer']
      joint_qvel_buffer = replay_buffer['joint_qvel_buffer']
      mocap_qpos_buffer = replay_buffer['mocap_qpos_buffer']
      cmd_buffer = replay_buffer['cmd_buffer']
      object_qpos_buffer = replay_buffer['object_qpos_buffer']
      # init scene
      for obj_name in operated_objects:
        qpos0 = object_qpos_buffer[obj_name][0]
        env.sim.data.set_joint_qpos(obj_name, qpos0)
      for mcp_name in operated_mocaps:
        qpos0 = mocap_qpos_buffer[mcp_name][0]
        env.sim.data.set_mocap_pos(mcp_name, qpos0[:3])
        env.sim.data.set_mocap_quat(mcp_name, qpos0[3:])
      for _ in range(10):  # step forward to drag mocap-controlled EE into place
        env.sim.step()
      print(">>> Scene reset to recorded initial state!")
      while True:
        env.render_extended()
        action = cmd_buffer[env.get_ts()]
        obs, reward, done, info = env.step(action)
        if env.get_ts() >= env._max_episode_steps: break

    # --- random wiggling (mostly debug)
    elif args.sim_mode == 'random':
      operated_joints = [
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
      operated_mocaps = ['robot0:mocap']
      while True:
        env.render_extended()
        for jnt_name in operated_joints:
          qvel = np.random.normal(loc=0.0, scale=2.0)
          env.sim.data.set_joint_qvel(jnt_name, qvel)
        for mcp_name in operated_mocaps:
          cur_pos = env.sim.data.get_mocap_pos(mcp_name)
          new_pos = cur_pos + np.random.normal(loc=0.0, scale=0.1, size=3)
          env.sim.data.set_mocap_pos(mcp_name, new_pos)
        env.ts += 1
        env.sim.step()
        if env.ts >= env._max_episode_steps: break

    # --- controller
    elif args.sim_mode == 'controller':
      # constants
      operated_joints = [  # <-- cmd_vel
          'robot0:shoulder_pan_joint',
          'robot0:shoulder_lift_joint',
          'robot0:upperarm_roll_joint',
          'robot0:elbow_flex_joint',
          'robot0:forearm_roll_joint',
          'robot0:wrist_flex_joint',
          'robot0:wrist_roll_joint',
      ]
      operated_mocaps = ['robot0:mocap']  # <-- cmd_ee
      operated_actuators = [  # <-- cmd_grp
          'robot0:r_gripper_finger_joint',
          'robot0:l_gripper_finger_joint',
      ]
      _h, _w = args.frame_res
      _cam_name = 'external_camera_1'
      # eval spec
      eval_spec = dict(episode_eval_spec)
      eval_spec['episode_id'] = episode_id
      eval_spec['init_goal_dist'] = _eval_goal_dist(env, env.task_object[0], env.task_goal[0])
      # command and trajectory information
      cmd_ee, cmd_grp = [], []
      traj_ee, traj_obj, pred_traj_ee, pred_traj_obj = [], [], [], []
      # reset
      predictor.reset()
      if args.goal_condition == 'target':
        target_frame = env.target_frame[0]
        predictor.set_goal(target_frame)
        # DEBUG dump target frame
        target_frame_path = os.path.join(wrk_dir, 'target-%05d.png' % (episode_id, ))
        # scipy.misc.imsave(target_frame_path, target_frame)
        Image.fromarray((target_frame * 255).astype(np.uint8)).save(target_frame_path)
      while True:
        # render frame
        if randomize_background:
          wall_geom_name = 'wall_04'
          wall_tex = texmodder.get_texture(wall_geom_name)
          frame_wall_tex = next(bg_video_cycler)
          img_frame_wall_tex = Image.fromarray(frame_wall_tex)
          img_frame_wall_tex = img_frame_wall_tex.resize((wall_tex.width, wall_tex.height))
          mod_wall_tex = np.array(img_frame_wall_tex)
          texmodder.set_rgb(wall_geom_name, mod_wall_tex)
        if args.observation_format == 'rgb':
          data = env.sim.render(width=_w, height=_h, camera_name=_cam_name)
          rgb_frame = np.copy(data[::-1, :, :])  # original image is upside-down
          recorders['rgb'].feed(rgb_frame)  # feed as uint8 frame to video recorder
          rgb_frame = rgb_frame / 255.0  # normalize RGB for predictor feeding
          obs_frame = rgb_frame
        elif args.observation_format == 'rgbd':
          rgb, depth = env.sim.render(width=_w, height=_h, camera_name=_cam_name, depth=True)
          rgb_frame = np.copy(rgb[::-1, :, :])  # original image is upside-down
          depth_frame = np.copy(depth[::-1])  # original image is upside-down
          recorders['rgb'].feed(rgb_frame)  # feed as uint8 frame to video recorder
          rgb_frame = rgb_frame / 255.0  # normalize RGB for predictor feeding
          obs_frame = np.concatenate([rgb_frame, np.expand_dims(depth_frame, axis=-1)], axis=-1)  # RGB-D
        # get robot state
        if args.controller == 'e2evmc':
          proprioception = np.zeros(shape=(7, ), dtype=np.float32)
          for idx_jnt, jnt_name in enumerate(operated_joints):
            proprioception[idx_jnt] = env.sim.data.get_joint_qpos(jnt_name)
        elif args.controller == 'vfs':
          mcp_name = 'robot0:mocap'
          proprioception = env.sim.data.get_mocap_pos(mcp_name)
        elif args.controller == 'tecnet':
          proprioception = np.zeros(shape=(10, ), dtype=np.float32)
          for idx_jnt, jnt_name in enumerate(operated_joints):
            proprioception[idx_jnt] = env.sim.data.get_joint_qpos(jnt_name)
          mcp_name = 'robot0:mocap'
          proprioception[-3:] = env.sim.data.get_mocap_pos(mcp_name)
        elif args.controller == 'static' or args.controller == 'gaussian':
          proprioception = np.zeros(shape=(7, ), dtype=np.float32)  # not used inside predictor
        # predict commands
        pred = predictor.predict(obs_frame, proprioception)
        action = np.concatenate([pred['cmd_ee'], pred['cmd_grp']])
        # feed additional debug output into separate recorders
        if args.debug:
          if predictor.cfg.proc_obs == 'dynimg':
            dynbuff_frame = (pred['dynbuff'] * 255.0).astype(np.uint8)
            recorders['dynbuff'].feed(dynbuff_frame)
          if predictor.cfg.proc_tgt == 'dyndiff':
            dyndiff_frame = (pred['dyndiff'] * 255.0).astype(np.uint8)
            recorders['dyndiff'].feed(dyndiff_frame)
        # print(action)  # DEBUG
        obs, reward, done, info = env.step(action)
        # perform eval checks
        goal_name = env.task_goal[0]
        obj_name = env.task_object[0]
        obj_vicinity = _eval_object_vicinity(env, obj_name)
        if obj_vicinity and eval_spec['obj_vicinity'] < 1:
          eval_spec['obj_vicinity'] += 1
          print(">>> Successfully reached %s!" % obj_name)
        grasp_success = _eval_grasp_success(env, obj_name)
        if grasp_success and eval_spec['grasp_success'] < 1:
          eval_spec['grasp_success'] += 1
          print(">>> Successfully grasped %s!" % obj_name)
        goal_dist = _eval_goal_dist(env, obj_name, goal_name)
        eval_spec['min_goal_dist'] = min([eval_spec['min_goal_dist'], goal_dist])
        eval_spec['max_goal_dist'] = max([eval_spec['max_goal_dist'], goal_dist])
        # record commands and trajectories
        cmd_ee.append(pred['cmd_ee'])
        cmd_grp.append(pred['cmd_grp'])
        traj_ee.append(np.copy(env.sim.data.get_mocap_pos('robot0:mocap')))
        traj_obj.append(np.copy(env.sim.data.get_site_xpos(env.task_object[0])))
        if 'pos_ee' in pred:
          pred_traj_ee.append(pred['pos_ee'])
        if 'pos_obj' in pred:
          pred_traj_obj.append(pred['pos_obj'])
        # terminate episode
        if env.get_ts() >= env._max_episode_steps: break
      # final evaluation of task success
      eval_spec['final_goal_dist'] = _eval_goal_dist(env, env.task_object[0], env.task_goal[0])
      task_success = _eval_task_success(env, obj_name, goal_name)
      if task_success and eval_spec['task_success'] < 1:
        eval_spec['task_success'] += 1
        print(">>> Successfully placed %s!" % obj_name)
      # save videos
      for rec_key in recorders.keys():
        video_path = recorders[rec_key].flush()
        if rec_key == 'rgb':
          eval_spec['video_file'] = video_path
      # TODO: currently broken, because of ffmpeg clash -> refacor and fix!
      # if args.debug:  # save command and trajectory videos
      #   # commands
      #   ani = _animate_commands(cmd_ee, cmd_grp)
      #   save_path = os.path.join(wrk_dir, 'commands_%06d.mp4' % episode_id)
      #   ani.save(save_path)
      #   # trajectories
      #   ani = _animate_trajectories(traj_ee, traj_obj, pred_traj_ee, pred_traj_obj)
      #   save_path = os.path.join(wrk_dir, 'trajectories_%06d.mp4' % episode_id)
      #   ani.save(save_path)
      # add eval spec to results
      eval_results.append(eval_spec)
      # print current success averages
      for k in ['obj_vicinity', 'grasp_success', 'task_success']:
        cur_avg = np.average([res[k] for res in eval_results]) * 100
        print(">>> Current average success rate for %s: %.02f" % (k, cur_avg))
      # append result to CSV
      writer.writerow(eval_spec)

    # --- unknown mode
    else:
      raise ValueError("Unknown simulation mode: %s" % (args.sim_mode, ))
  
  # --- end main loop, cleanup
  if args.sim_mode == 'controller':
    csv_report.close()
    txt_report_file = os.path.join(args.wrk_dir, 'controller', 'final_results.txt')
    with open(txt_report_file, 'w') as fp:
      for k in ['obj_vicinity', 'grasp_success', 'task_success']:
        cur_avg = np.average([res[k] for res in eval_results]) * 100
        fp.write(f"{k}\t{cur_avg:.2f}\n")

#endregion

# ---------- program entry point ----------

if __name__ == "__main__":
  ARGS, UNPARSED = ARGPARSER.parse_known_args()
  main(ARGS)
