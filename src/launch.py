import sys

sys.path.append("/usr/lib/python3/dist-packages")
import rospy

import argparse
import os

from tycho_demo import run_demo
from tycho_demo.addon import add_visualize_function, add_replay_pose_function, add_tuning_function, add_recording_function, add_grabbing_function, add_logging_function
from diffusion import add_diffusion_function


def get_args():
  parser = argparse.ArgumentParser()
  dir_path = os.path.dirname(os.path.realpath(__file__))
  parser.add_argument("model_name")
  parser.add_argument("-p", "--models_path", default=os.path.join(dir_path, "..", "models"))
  parser.add_argument("-s", "--save_recording_path", default=os.path.join(dir_path, '..', 'recording'))
  return parser.parse_args()

def register_modes(state):
  add_visualize_function(state)
  add_replay_pose_function(state)
  add_diffusion_function(state)
  add_tuning_function(state)
  add_recording_function(state)
  add_grabbing_function(state)
  add_logging_function(state)


if __name__ == "__main__":
  args = get_args()
  rospy.init_node("tycho_imitation")
  run_demo(callback_func=register_modes,
      params={'save_record_folder': args.save_recording_path,
              'modelsPath': args.models_path,
              'model_name': args.model_name},
      cmd_freq=20)
