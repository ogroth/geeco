"""
Contains helper functions for loading and saving hyperparameters and configs
of main() run scripts.
"""

import os
import argparse
import time
import datetime
import json


def save_run_command(argparser: argparse.ArgumentParser, run_dir):
  """
  Saves the CLI run command from the argument parser into the run directory in
  JSON format.
  
  :return run_cmd_path: path to the saved JSON file
  """
  ts = time.time()
  ts_str = datetime.datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S%f')[:-3]  # including ms
  parsed, unparsed = argparser.parse_known_args()
  run_cmd = {}
  run_cmd['parsed_args'] = parsed.__dict__
  run_cmd['unparsed_args'] = unparsed
  fn = "%s-runcmd.json" % (ts_str, )
  run_cmd_path = os.path.join(run_dir, fn)
  with open(run_cmd_path, 'w') as f:
    json.dump(run_cmd, f, indent=2, sort_keys=True)
  return run_cmd_path
