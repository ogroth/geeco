"""
Run script for a pushing Gym environment featuring a Fetch robot.
Can be used to collect expert demonstrations, replay recorded trajectories or
run and evaluate a controller.

Example calls:

* Simulation of expert trajectories with onscreen rendering:
LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libGLEW.so" python3 gym_pushing.py \
  --rendering_mode viewer

* Simulation of expert trajectories with video rendering:
python3 gym_pushing.py \
  --rendering_mode video

* Running a controller:
python3 gym_pushing.py \
  --shapes push-pad1-cube1 \
  --sim_mode controller \
  --model_dir ../models/geeco-f/push11 \
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
import numpy as np
from PIL import Image
from tqdm import tqdm

from data.geeco_gym import load_target_frame
from mj_engine.engine.recorder import MjVideoRecorder
from models.e2evmc.predictor import E2EVMCPredictor, GoalE2EVMCPredictor
from utils.runscript import save_run_command


#region: CLI arguments

ARGPARSER = argparse.ArgumentParser(
    description='Collect data for a simple block pushing task with a Fetch robot in a\
      gym environment.')
ARGPARSER.add_argument(
    '--wrk_dir', type=str, default='../logs/gym_pushing',
    help="The path to the working directory for this run (where logs, videos \
      etc. will be stored).")
ARGPARSER.add_argument(
    '--shapes', type=str, default='push-pad2-cube2',
    help="The shape set used for pushing. \
      Available: push-pad1-cube1 | push-pad1-cube2 | push-pad2-cube1 | push-pad2-cube2")
ARGPARSER.add_argument(
    '--sim_mode', type=str, default='collect',
    help="The simulation mode to use: \
      collect = collect expert data \
      replay = replay from control buffer \
      random = random freestyle control \
      controller = run with controller")
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
    '--replay_buffer', type=str, default='../data/gym-push-pad2-cube2-v4/data/replay_buffer_0001.pkl',
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
      target = target image provided, conditional reflex")
ARGPARSER.add_argument(
    '--model_dir', type=str, default='../tmp/models/geeco-f/push11',
    help="Directory from where the controller model is loaded.")
ARGPARSER.add_argument(
    '--checkpoint_name', type=str, default=None,
    help="Specific checkpoint to load. If none, load latest in model_dir.")
ARGPARSER.add_argument(
    '--dataset_dir', type=str, default='../data/gym-push-pad2-cube2-v4/',
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

#endregion


#region: constants

NAME_GRIPPER = 'robot0:grip'
NAME_TABLE = 'table0'
OFFSET_PRE_PUSH = 0.1  # gripper offset above target object
DIST_PRE_PUSH = 0.015  # distance to target object for pre-push pose
DIST_PUSH = 0.002  # distance to target object for push pose (defines firmness of grip)
DIST_GOAL = 0.01  # radius around goal pos, positions within are considered 'in goal'
MULT_POS_ACTION = 6.0  # multiplier for pos action (for faster movement to target pos)
TOL_GRIPPER_RELEASE = 0.0001  # tolerance between gripper state and full open (for release)
PAUSE_AFTER_DROP = 10  # number of simulation steps for the robot to sit idle after placing an object (to let the stack settle)
CMD_GRIPPER_OPEN = 1.0
CMD_GRIPPER_CLOSE = -1.0
CMD_GRIPPER_NOOP = 0.0

GOAL_NAMES = {  # env_name -> [goal_name]
    'push-pad1-cube1' : ['goal0'],
    'push-pad1-cube2' : ['goal0'],
    'push-pad2-cube1' : ['goal0', 'goal1'],
    'push-pad2-cube2' : ['goal0', 'goal1'],
}
CUBE_NAMES = {  # env_name -> [cube_name]
    'push-pad1-cube1' : ['object0'],
    'push-pad1-cube2' : ['object0', 'object1'],
    'push-pad2-cube1' : ['object0'],
    'push-pad2-cube2' : ['object0', 'object1'],
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
      task_dict = {'goal' : task_row[0], 'object' : task_row[1]}
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
      task_dict = {'goal' : task_row[0], 'object' : task_row[1]}
      reset_state = {
          'init' : state_dict,
          'task' : task_dict,
          'target' : load_target_frame(dataset_dir, record_name, load_depth=False),  # HOTFIX: disabled depth!
          'robot' : robot_dict,
      }
      env.enqueue_reset_state(reset_state)

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

def _move_to_pre_push_x(env, obj_name: str):
  """Move gripper into pre-push_x pose for object."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # first make the gripper go in front of the object
  object_oriented_goal = object_rel_pos.copy()
  object_oriented_goal[0] -= OFFSET_PRE_PUSH
  # set goal
  env.sample_goal(object_oriented_goal)
  # movement loop
  while np.linalg.norm(object_oriented_goal) >= DIST_PRE_PUSH \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[0] -= OFFSET_PRE_PUSH
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(object_oriented_goal)):
      action[i] = object_oriented_goal[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # pre-push pose reached

def _push_x(env, obj_name: str, goal_pos):
  """Push object to a goal position in x direction."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # set goal
  goal_pos_x = object_pos.copy()
  goal_pos_x[0] = goal_pos[0]
  env.sample_goal(goal_pos_x)
  # movement loop
  while np.linalg.norm(goal_pos_x - object_pos) >= DIST_GOAL \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(goal_pos_x - object_pos)):
      action[i] = (goal_pos_x - object_pos)[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # object has been moved to goal position

def _move_to_pre_push_y(env, obj_name: str, goal_pos):
  """Move gripper into pre-push_y pose for object."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # first make the gripper go to the side of the object
  # decide which side to go
  object_oriented_goal = object_rel_pos.copy()
  offset_sign = 1
  if goal_pos[1] - object_pos[1] > 0:
    offset_sign = -1
  object_oriented_goal[1] += offset_sign * OFFSET_PRE_PUSH
  # set goal
  env.sample_goal(object_oriented_goal)
  # move back a little bit
  for _ in range(0, 3):
    # adjust action
    action = [0, 0, 0, 0]
    action[0] = -OFFSET_PRE_PUSH * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  # movement loop
  while np.linalg.norm(object_oriented_goal) >= DIST_PRE_PUSH \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    object_oriented_goal = object_rel_pos.copy()
    object_oriented_goal[1] += offset_sign * OFFSET_PRE_PUSH
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(object_oriented_goal)):
      action[i] = object_oriented_goal[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # pre-push pose reached

def _push_y(env, obj_name: str, goal_pos):
  """Push object to a goal position in y direction."""
  # initial positions
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  object_rel_pos = object_pos - grip_pos
  # set goal
  goal_pos_y = object_pos.copy()
  goal_pos_y[1] = goal_pos[1]
  env.sample_goal(goal_pos_y)
  # movement loop
  while np.linalg.norm(goal_pos_y - object_pos) >= DIST_GOAL \
    and env.ts < env._max_episode_steps:
    env.render_extended()
    # adjust action
    action = [0, 0, 0, 0]
    for i in range(len(goal_pos_y - object_pos)):
      action[i] = (goal_pos_y - object_pos)[i] * MULT_POS_ACTION
    action[-1] = CMD_GRIPPER_CLOSE
    # forward environment
    obs, reward, done, info = env.step(action)
    # re-compute positions
    grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
    object_pos = env.sim.data.get_site_xpos(obj_name)
    object_rel_pos = object_pos - grip_pos
  return True  # object has been moved to goal position

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

def _check_obj_on_target_y(env, obj_name: str, goal_pos):
  object_pos = env.sim.data.get_site_xpos(obj_name)
  return np.linalg.norm(goal_pos[1] - object_pos[1]) < DIST_GOAL

def _push(env, obj_names, goal_name):
  for obj_idx, obj_name in enumerate(obj_names):
    _move_to_pre_push_x(env, obj_name)
    goal = env.sim.data.get_site_xpos(goal_name)
    _push_x(env, obj_name, goal)
    if not _check_obj_on_target_y(env, obj_name, goal):
      _move_to_pre_push_y(env, obj_name, goal)
      _push_y(env, obj_name, goal)
    _idle(env, PAUSE_AFTER_DROP)
  while env.ts < env._max_episode_steps:
    env.render_extended()
    action = [0, 0, 0, 0]
    obs, reward, done, info = env.step(action)

#endregion


#region: sub-routines for controller evaluation

# TODO: refactor into application constants / make parameter
OBJ_VICINITY = 0.0625  # radius around manipulated object; determines vicinity test
GOAL_VICINITY = 0.05  # radius around goal object; determines task success test

def _eval_object_vicinity(env, obj_name: str):
  """Checks whether the gripper has come close enough to the object."""
  grip_pos = env.sim.data.get_site_xpos(NAME_GRIPPER)
  object_pos = env.sim.data.get_site_xpos(obj_name)
  dist = np.linalg.norm(object_pos - grip_pos)
  return dist <= OBJ_VICINITY

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
    # TODO: refactor hard-coded constants
    if args.sim_mode == 'controller':  # max_episode_steps depend
      max_episode_steps = 200  # normal controller execution (2x expert demonstration time)
    else:
      max_episode_steps = 100  # time for expert demonstration
    register(
        id='PushingEnv{}-v1'.format(suffix),
        entry_point='geeco_gym:PushingEnv',
        kwargs=kwargs,
        max_episode_steps=max_episode_steps,
    )

  # --- build gym environment
  env = gym.make('PushingEnv-v1')
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
    recorder = MjVideoRecorder(
        ctx_name='default', ctx_type='rgb', cam_name='cam_default',
        record_name='test', record_dir=wrk_dir,
        res_height=256, res_width=256)

  # --- set up evaluation data structures
  if args.sim_mode == 'controller':
    eval_results = []
    episode_eval_spec = [
        ('episode_id', 0),
        ('obj_vicinity', 0),  # binary success indicator
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
      obj_names = [env.task_object]
      goal_name = env.task_goal
      # save meta information
      meta_info_dict = env.encoding_meta._asdict()
      meta_info_path = os.path.join(wrk_dir, 'meta_info.json')
      with open(meta_info_path, 'w') as fp:
        json.dump(meta_info_dict, fp, indent=2, sort_keys=True)
      # start tfrecorder
      if args.rendering_mode == 'tfrecord':
        record_name = 'replay_buffer_%04d' % (episode_id, )
        env.start_tfrecorder(record_name)
      # perform pushing
      _push(env, obj_names=obj_names, goal_name=goal_name)
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
        # print(env.ts, action)  # DEBUG
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
      eval_spec['init_goal_dist'] = _eval_goal_dist(env, env.task_object, env.task_goal)
      # reset
      # env.ts = 0
      predictor.reset()
      if args.goal_condition == 'target':
        target_frame = env.target_frame
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
          recorder.feed(rgb_frame)  # feed as uint8 frame to video recorder
          rgb_frame = rgb_frame / 255.0  # normalize RGB for predictor feeding
          obs_frame = rgb_frame
        elif args.observation_format == 'rgbd':
          rgb, depth = env.sim.render(width=_w, height=_h, camera_name=_cam_name, depth=True)
          rgb_frame = np.copy(rgb[::-1, :, :])  # original image is upside-down
          depth_frame = np.copy(depth[::-1])  # original image is upside-down
          recorder.feed(rgb_frame)  # feed as uint8 frame to video recorder
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
        # print(action)  # DEBUG
        obs, reward, done, info = env.step(action)
        # perform eval checks
        goal_name = env.task_goal
        obj_name = env.task_object
        obj_vicinity = _eval_object_vicinity(env, obj_name)
        if obj_vicinity and eval_spec['obj_vicinity'] < 1:
          eval_spec['obj_vicinity'] += 1
          print(">>> Successfully reached %s!" % obj_name)
        goal_dist = _eval_goal_dist(env, obj_name, goal_name)
        eval_spec['min_goal_dist'] = min([eval_spec['min_goal_dist'], goal_dist])
        eval_spec['max_goal_dist'] = max([eval_spec['max_goal_dist'], goal_dist])
        # terminate episode
        if env.get_ts() >= env._max_episode_steps: break
      # final evaluation of task success
      eval_spec['final_goal_dist'] = _eval_goal_dist(env, env.task_object, env.task_goal)
      task_success = _eval_task_success(env, obj_name, goal_name)
      if task_success and eval_spec['task_success'] < 1:
        eval_spec['task_success'] += 1
        print(">>> Successfully pushed %s!" % obj_name)
      # save as video
      video_path = recorder.flush()
      eval_spec['video_file'] = video_path
      # add eval spec to results
      eval_results.append(eval_spec)
      # print current success averages
      for k in ['obj_vicinity', 'task_success']:
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
      for k in ['obj_vicinity', 'task_success']:
        cur_avg = np.average([res[k] for res in eval_results]) * 100
        fp.write(f"{k}\t{cur_avg:.2f}\n")

#endregion

# ---------- program entry point ----------

if __name__ == "__main__":
  ARGS, UNPARSED = ARGPARSER.parse_known_args()
  main(ARGS)
