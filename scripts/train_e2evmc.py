"""Training script for End-to-end visuomotor controllers."""

import argparse
import os
from stat import ST_CTIME
import shutil
import re
import json
import pprint

import tensorflow as tf

from data.geeco_gym import pickplace_input_fn
from models.e2evmc.estimator import e2evmc_model_fn, goal_e2evmc_model_fn
from models.e2evmc.params import create_e2evmc_config
from models.e2evmc.utils import save_model_config, load_model_config
from utils.runscript import save_run_command


# ---------- command line arguments ----------

ARGPARSER = argparse.ArgumentParser(description='Train E2E VMC.')
# --- directory parameters
ARGPARSER.add_argument(
    '--dataset_dir', type=str, default='../data/gym-pick-pad2-cube2-v4',
    help='The path to the dataset (needs to conform with gym_provider).')
ARGPARSER.add_argument(
    '--split_name', type=str, default='default',
    help='The name of the data split to be used.')
ARGPARSER.add_argument(
    '--model_dir', type=str, default='../tmp/models/geeco-f',
    help='The directory where the model will be stored.')
# --- model parameters
ARGPARSER.add_argument(
    '--observation_format', type=str, default='rgb',
    help='Observation data to be used (sets img_channels): rgb | rgbd.')
ARGPARSER.add_argument(
    '--control_mode', type=str, default='cartesian',
    help='Control mode of the robot: cartesian | velocity.')
ARGPARSER.add_argument(
    '--goal_condition', type=str, default='none',
    help='Conditioning mode of the reflex. Options are: none | target. \
      none = no goal provided, unconditional reflex \
      target = target image provided, conditional reflex')
ARGPARSER.add_argument(
    '--window_size', type=int, default=4,
    help='The number of frames to process before making prediction.')
ARGPARSER.add_argument(
    '--dim_h_lstm', type=int, default=128,
    help='Hidden state dimension of the LSTM.')
ARGPARSER.add_argument(
    '--dim_h_fc', type=int, default=128,
    help='Output dimension of the LSTM (before decoding heads).')
ARGPARSER.add_argument(
    '--dim_s_obs', type=int, default=256,
    help='Output dimension of the observation encoding.')
ARGPARSER.add_argument(
    '--dim_s_dyn', type=int, default=256,
    help='Output dimension of the dynamics encoding.')
ARGPARSER.add_argument(
    '--dim_s_diff', type=int, default=256,
    help='Output dimension of the target difference encoding.')
ARGPARSER.add_argument(
    '--proc_obs', type=str, default='sequence',
    help='The processing type of the frame buffer: sequence | dynimg')
ARGPARSER.add_argument(
    '--proc_tgt', type=str, default='constant',
    help='The processing type of the target frame: constant | residual | dyndiff')
ARGPARSER.add_argument(
    '--l2_regularizer', type=float, default=0.0,
    help='The weight of the L2 weight regularizer. Zero disables weight regularization.')
ARGPARSER.add_argument(
    '--lambda_aux', type=float, default=1.0,
    help='The weight of the auxiliary pose prediction losses. Zero disables them.')
# --- data parameters
ARGPARSER.add_argument(
    '--data_encoding', type=str, default='v4',
    help='Version of the data encoding. Available: v1 | v2 | v3 | v4')
# --- training parameters
ARGPARSER.add_argument(
    '--lr', type=float, default=1e-4,
    help='The learning rate of the ADAM solver.')
ARGPARSER.add_argument(
    '--train_epochs', type=int, default=10,
    help='The number of epochs to train.')
# --- snapshot management
ARGPARSER.add_argument(
    '--ckpt_steps', type=int, default=10000,
    help='Number of steps between checkpoint saves.')
ARGPARSER.add_argument(
    '--num_last_ckpt', type=int, default=2,
    help='Number of last snapshots to keep.')
ARGPARSER.add_argument(
    '--num_best_ckpt', type=int, default=5,
    help='Number of best performing snapshots to keep.')
# --- memory management
ARGPARSER.add_argument(
    '--batch_size', type=int, default=32,
    help='The number of data points per batch. Each data points contains \
      `window_size` number of annotated frames as well as a tensor containing \
      the target commands.')
ARGPARSER.add_argument(
    '--memcap', type=float, default=0.8,
    help='Maximum fraction of memory to allocate per GPU.')
ARGPARSER.add_argument(
    '--num_threads', type=int, default=4,
    help='How many parallel threads to run for data fetching.')
ARGPARSER.add_argument(
    '--prefetch_size', type=int, default=4,
    help='How many batches to prefetch.')
ARGPARSER.add_argument(
    '--shuffle_buffer', type=int, default=64,
    help='Number of shuffled examples to draw minibatch from.')
# --- logging
ARGPARSER.add_argument(
    '--log_steps', type=int, default=1000,
    help='Global steps between log output.')
ARGPARSER.add_argument(
    '--debug', default=False, action='store_true',
    help="Enables debugging mode for more verbose logging and tensorboard \
      output.")
ARGPARSER.add_argument(
    '--initial_eval', default=False, action='store_true',
    help="Runs an evaluation before the first training iteration.")


# ---------- constants ----------

_OBSERVATION_FORMAT_TO_CHANNELS = {
    'rgb' : 3,
    'rgbd' : 4,
}

_GOAL_CONDITION_TO_MODEL = {  # cond -> (model_fn, model_scope)
    'none' : (e2evmc_model_fn, 'VMC'),
    'target' : (goal_e2evmc_model_fn, 'GoalVMC'),
}


# ---------- helper_fn ----------

# TODO: refactor into general training utils
def _export_snapshot(model_dir, eval_results, num_best_ckpt):
  """
  Manages `num_best_ckpt` snapshots of the best performing models according to
  eval_results['loss'] in <model_dir>/snapshots/
  """
  # --- directory setup
  snapshots_dir = os.path.join(model_dir, 'snapshots')
  os.makedirs(snapshots_dir, exist_ok=True)
  # --- snapshot index
  snapshot_index_file = os.path.join(snapshots_dir, 'snapshot_index.json')
  if os.path.exists(snapshot_index_file):  # load snapshot index
    with open(snapshot_index_file, 'r') as fp:
      snapshot_index = json.load(fp)
  else:  # create new snapshot index
    snapshot_index = {}  # <snapshot_name> -> {step, loss, dir}
  print(">>> Current snapshot index contains %d entries." % len(snapshot_index))  # DEBUG
  # --- get latest runcmd, config, step and loss
  ckpt_name = os.path.basename(tf.train.latest_checkpoint(model_dir))
  step = int(re.search(r'\d+', ckpt_name).group(0))
  loss = float(eval_results['loss'])
  runcmd_files = [fn for fn in os.listdir(model_dir) if fn.endswith('runcmd.json')]
  runcmd_files = [os.path.join(model_dir, fn) for fn in runcmd_files]
  runcmd_files = [(fn, os.stat(fn)[ST_CTIME]) for fn in runcmd_files]
  runcmd_files.sort(key=lambda t: t[1])  # sort by creation date asc
  runcmd_path = runcmd_files[-1][0]  # get latest ending in _runcmd.json
  config_files = [fn for fn in os.listdir(model_dir) if fn.endswith('config.json')]
  config_files = [os.path.join(model_dir, fn) for fn in config_files]
  config_files = [(fn, os.stat(fn)[ST_CTIME]) for fn in config_files]
  config_files.sort(key=lambda t: t[1])  # sort by creation date asc
  config_path = config_files[-1][0]  # get latest ending in _config.json
  # --- export current checkpoint to <model_dir>/snapshots/<ckpt_name>
  ckpt_dir = os.path.join(snapshots_dir, ckpt_name)
  os.makedirs(ckpt_dir, exist_ok=True)
  for cfg_fn in [runcmd_path, config_path]:  # copy configs
    shutil.copy(src=cfg_fn, dst=ckpt_dir)
  for ckpt_fn in [fn for fn in os.listdir(model_dir) if fn.startswith(ckpt_name)]:  # checkpoint files
    shutil.copy(src=os.path.join(model_dir, ckpt_fn), dst=ckpt_dir)
  with open(os.path.join(ckpt_dir, 'checkpoint'), 'w') as fp:  # create checkpoint header
    fp.write("model_checkpoint_path: \"%s\"\n" % ckpt_name)
  print(">>> Exported current checkpoint (step=%d; loss=%.06f) to %s." % (step, loss, ckpt_dir))  # DEBUG
  pprint.pprint(os.listdir(ckpt_dir))  # DEBUG
  # --- update index
  snapshot_index[ckpt_name] = {
      'step' : step,
      'loss' : loss,
      'dir' : ckpt_dir,
  }
  # --- gc worst snapshot, if save slots are exceeded
  if len(snapshot_index) > num_best_ckpt:
    ckpt_by_loss = list(snapshot_index.items())
    ckpt_by_loss.sort(key=lambda t: t[1]['loss'])
    worst_ckpt, _ = ckpt_by_loss[-1]  # get worst checkpoint name
    worst_ckpt_dir = snapshot_index[worst_ckpt]['dir']
    shutil.rmtree(worst_ckpt_dir)
    worst_info = snapshot_index.pop(worst_ckpt)
    print(">>> Removed worst snapshot (step=%d; loss=%.06f): %s" % \
      (worst_info['step'], worst_info['loss'], worst_info['dir']))  # DEBUG
  # --- save snapshot index
  pprint.pprint(snapshot_index)  # DEBUG
  with open(snapshot_index_file, 'w') as fp:
    json.dump(snapshot_index, fp, indent=2, sort_keys=True)
  print(">>> Saved snapshot index: %s" % snapshot_index_file)
  return ckpt_dir


# ---------- main ----------

def main(args):
  """Executes the model training."""
  # --- directory setup
  os.makedirs(name=args.model_dir, exist_ok=True)
  # --- run command
  run_cmd_path = save_run_command(argparser=ARGPARSER, run_dir=args.model_dir)
  # --- run config
  gpu_options = tf.GPUOptions(
      allow_growth=True,
      per_process_gpu_memory_fraction=args.memcap)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  run_config = tf.estimator.RunConfig(
      session_config=sess_config,
      save_checkpoints_steps=args.ckpt_steps,
      keep_checkpoint_max=args.num_last_ckpt)
  # --- model config, TODO: refactor config CLI interface, loading and merging into model utils
  config_name = 'e2evmc_config'  # TODO: add config name CLI parameter
  config_fn = "%s.json" % (config_name, )
  config_path = os.path.join(args.model_dir, config_fn)
  if os.path.exists(config_path):  # load model config from previous run
    custom_params = load_model_config(args.model_dir, config_name)
    e2evmc_config = create_e2evmc_config(custom_params)
    print(">>> Loaded existing model config from %s" % (config_path, ))
  else:  # create new model config from CLI parameters
    custom_params = {
        'img_channels' : _OBSERVATION_FORMAT_TO_CHANNELS[args.observation_format],
        'control_mode' : args.control_mode,
        'window_size' : args.window_size,
        'dim_h_lstm' : args.dim_h_lstm,
        'dim_h_fc' : args.dim_h_fc,
        'dim_s_obs' : args.dim_s_obs,
        'dim_s_dyn' : args.dim_s_dyn,
        'dim_s_diff' : args.dim_s_diff,
        'proc_obs' : args.proc_obs,
        'proc_tgt' : args.proc_tgt,
        'l2_regularizer' : args.l2_regularizer,
        'lambda_aux' : args.lambda_aux,
        'batch_size' : args.batch_size,
        'lr' : args.lr,
    }
    e2evmc_config = create_e2evmc_config(custom_params)
    save_model_config(e2evmc_config._asdict(), args.model_dir, config_name)
    print(">>> Saved model config to %s" % (config_path, ))
  # --- estimator setup
  estimator_params = {
      'e2evmc_config' : e2evmc_config,
      'log_steps' : args.log_steps,
      'debug' : args.debug,
  }
  model_fn, model_scope = _GOAL_CONDITION_TO_MODEL[args.goal_condition]
  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      model_dir=args.model_dir,
      config=run_config,
      params=estimator_params)
  # --- data pipeline
  input_fn = lambda estimator_mode: pickplace_input_fn(
      dataset_dir=args.dataset_dir,
      split_name=args.split_name,
      mode=estimator_mode,
      encoding=args.data_encoding,
      window_size=e2evmc_config.window_size,
      fetch_target=(args.goal_condition == 'target'),
      shuffle_buffer=args.shuffle_buffer,
      batch_size=args.batch_size,
      num_epochs=1,
      num_threads=args.num_threads,
      prefetch_size=args.prefetch_size,
      seed=None)
  train_input = lambda: input_fn(estimator_mode='train')
  eval_input = lambda: input_fn(estimator_mode='eval')

  # --- extended init
  if args.initial_eval:
    eval_results = estimator.evaluate(input_fn=eval_input)
    # _export_snapshot(args.model_dir, eval_results, args.num_best_ckpt)

  # --- main training loop
  for epoch_id in range(args.train_epochs):
    estimator.train(input_fn=train_input)
    eval_results = estimator.evaluate(input_fn=eval_input)
    _export_snapshot(args.model_dir, eval_results, args.num_best_ckpt)


if __name__ == '__main__':
  print(">>> Training E2E VMC.")
  PARSED_ARGS, UNPARSED_ARGS = ARGPARSER.parse_known_args()
  print(">>> PARSED ARGV:")
  pprint.pprint(PARSED_ARGS)
  print(">>> UNPARSED ARGV:")
  pprint.pprint(UNPARSED_ARGS)
  tf.logging.set_verbosity(tf.logging.INFO)
  main(PARSED_ARGS)
