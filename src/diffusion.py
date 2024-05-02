import os
from threading import Lock, Thread, Event
import time
import numpy as np


import rospy
from geometry_msgs.msg import PointStamped, PoseStamped

from scipy.spatial.transform import Rotation as scipyR

from diffusion_policy.policy import DiffusionPolicy
from utils import load_yaml_config

from tycho_env.utils import R_optitrack2base, print_and_cr
from tycho_env.utils import construct_choppose, get_ee_from_tracker
from tycho_demo.utils import ChopPosePublisher
from tycho_demo.addon.recording import set_rosbag_recording
from tycho_demo.addon.logger import _press_logging
from tycho_demo.demo_interface import ROBOT_FEEDBACK_FREQUENCY
from tycho_infra.mujoco_assets.TychoPhysics import TychoPhysics
from tycho_infra.mujoco_assets.MujocoIK import get_IK_from_mujoco

# Singleton
CHOPPOSE_PUBLISHER = ChopPosePublisher('/Choppose')
TARGET_CHOPPOSE_PUBLISHER = ChopPosePublisher('/Choppose_target')

def add_diffusion_function(state):
  state.model_folder = os.path.join(state.params["modelsPath"], state.params["model_name"])
  state._diffusion_lock = Lock()
  state.diffusion_state = {}
  state.diffusion_task = None
  state.stop_inference_event = Event()
  load_agent(state)

  state.handlers['i'] = state.handlers['I'] = _diffusion
  state.handlers[';'] = _reload_agent
  state.handlers[':'] = _print_agent

  state.modes['diffusion'] = __diffusion

  setup_diffusion(state)

  state.sim = TychoPhysics([-1.93332, 2.10996, 2.33660, 0.23878, 1.20642, 0.00378, -0.37602])

def _reload_agent(key, state):
  print_and_cr("Reloading policy")
  load_agent(state)

def _print_agent(key, state):
  print_and_cr(f"Current agent: {state.config.policy_pickle_fn}")

def _diffusion(key, state):
  if state.mode != "diffusion":
    stop_inference_task(state)
    init_diffusion_state(state)
    start_inference_task(state)
    state.time = time.time()

    state.mode = 'diffusion'
    if key.islower():
      set_rosbag_recording(state, True)
      _press_logging("l", state)
      state.tracked_objs.update({
        "choppose_target": np.zeros(8)
      })
    print_and_cr("Entering diffusion mode")
  else:
    set_rosbag_recording(state, False)
    if state.is_logging_to:
      _press_logging("l", state)

def start_inference_task(state):
  stop_inference_task(state)
  state.diffusion_task = Thread(target=inference_task, args=(state,))
  state.diffusion_task.start()

def stop_inference_task(state):
  if state.diffusion_task is not None:
    state.stop_inference_event.set()
    state.diffusion_task.join()
    state.diffusion_task = None
    state.stop_inference_event.clear()
    state.diffusion_agent.clear_obs()

def inference_task(state):
  Ta = state.diffusion_agent.config.action_horizon
  control_freq = ROBOT_FEEDBACK_FREQUENCY / state.counter_skip_freq
  inference_freq = control_freq / Ta
  loop_period = 1. / inference_freq
  print_and_cr(f"loop freq={inference_freq}hz, period={loop_period:.3f}s")
  while not state.stop_inference_event.is_set():
    start = time.perf_counter()
    actions = state.diffusion_agent.get_action()
    if actions is not None:
      with state._diffusion_lock:
        state.diffusion_state["pred_action"] = actions
        state.diffusion_state["action_idx"] = 0
    else:
      print_and_cr("WARN: No observations available!")
    elapsed = time.perf_counter() - start
    if elapsed < loop_period:
      if state.stop_inference_event.wait(timeout=loop_period - elapsed):
        break
    else:
      print_and_cr("WARN: Inference loop overrun!")
  state.stop_inference_event.clear()

def load_agent(state):
  stop_inference_task(state)
  with state._diffusion_lock:
    print_and_cr(f"Loading diffusion policy from {state.model_folder}")
    config = load_yaml_config(model_folder=state.model_folder)
    state.config = config
    agent = DiffusionPolicy.load(config.policy_pickle_fn)
    state.diffusion_config = agent.config
    state.diffusion_agent = agent

def setup_diffusion(state):
  print_and_cr('Create ROS subscriber and publisher for diffusion')
  if not state.config.data.no_ball:
    def callback(pointstamped_msg):
      _p = pointstamped_msg.point
      raw_point = np.array([_p.x, _p.y, _p.z, 1])
      with state._diffusion_lock:
        state.diffusion_state['ball'] = R_optitrack2base.dot(raw_point)[0:3].flatten()
    topic = "/Ball/point"
    rospy.Subscriber(topic, PointStamped, callback)
    print_and_cr(f"Waiting for messages on {topic}")
    rospy.wait_for_message(topic, PointStamped, timeout=1.0)
    print_and_cr("Done!")
  if state.config.data.use_target_pose:
    def target_pose_callback(pose_msg: PoseStamped):
      p = pose_msg.pose.position
      q = pose_msg.pose.orientation
      pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w])
      with state._diffusion_lock:
        state.diffusion_state['target_pose'] = pose
    dconfig = state.config.data
    topic = dconfig.target_pose_topic if dconfig.target_pose_topic else "/target/pose"
    rospy.Subscriber(topic, PoseStamped, target_pose_callback, queue_size=10)
    print_and_cr(f"Waiting for messages on {topic}")
    rospy.wait_for_message(topic, PoseStamped, timeout=1.0)
    print_and_cr("Done!")
  
  # warm start diffusion policy
  print_and_cr("Warm starting...")
  state.diffusion_agent.get_action(gen_state(state))
  print_and_cr("Done!")

def init_diffusion_state(state):
  state.last_diffusion_cmd = state.current_position
  with state._diffusion_lock:
    state.diffusion_state["action_idx"] = 0
    state.diffusion_state["pred_action"] = None

def xyzquat_to_trf(xyzquat):
  assert len(xyzquat) == 7
  trf = np.eye(4)
  trf[:3,:3] = scipyR.from_quat(xyzquat[3:]).as_matrix()
  trf[:3,3] = xyzquat[:3]
  return trf

def gen_state(state):
  current_choppose = construct_choppose(state.arm, state.current_position)

  if "piggy_bank" in state.config.model_name:
    xyz, quat, chop = current_choppose[:3], current_choppose[3:7], current_choppose[7:8]
    rpy = scipyR.from_quat(quat).as_euler("xyz")
    current_choppose = np.hstack([xyz, rpy[1:2], chop])

  diffusion_state = [current_choppose]
  with state._diffusion_lock:
    if not state.config.data.no_ball:
      diffusion_state.append(np.array(state.diffusion_state['ball']))
    if state.config.data.use_target_pose:
      pose = state.diffusion_state["target_pose"].copy()
      if "piggy_bank" in state.config.model_name:
        ee_pose = construct_choppose(state.arm, state.current_position)[:7]
        if "prev_tp" in state.diffusion_state:
          if np.all(state.diffusion_state["target_pose"] == state.diffusion_state["prev_tp"]):
            print_and_cr("WARN: Lost coin!")
            prev_ee_trf = xyzquat_to_trf(state.diffusion_state["prev_ee"])
            prev_coin_trf = xyzquat_to_trf(state.diffusion_state["prev_tp"])
            ee_trf = xyzquat_to_trf(ee_pose)
            interp_coin_trf = ee_trf @ np.linalg.inv(prev_ee_trf) @ prev_coin_trf
            state.diffusion_state["target_pose"][0:3] = interp_coin_trf[:3,3]
            state.diffusion_state["target_pose"][3:7] = scipyR.from_matrix(interp_coin_trf[:3,:3]).as_quat()
        state.diffusion_state["prev_ee"] = ee_pose.copy()
        state.diffusion_state["prev_tp"] = state.diffusion_state["target_pose"].copy()

        coin_quat = state.diffusion_state["target_pose"][3:7]

        seq = "XYZ"
        coin_angles = scipyR.from_quat(coin_quat).as_euler(seq)
        diffusion_state.append(state.diffusion_state["target_pose"][:3])
        diffusion_state.append(coin_angles[:2])
      else:
        diffusion_info["target_pose"] = state.diffusion_state["target_pose"].copy()
        diffusion_state.append(diffusion_info["target_pose"])
  state = {"state": np.hstack(diffusion_state).flatten()}
  return state

def chopify_piggy_bank(state, only_ee):
  xyz, pitch, rest = state[:3], state[3], state[4:]
  rpy = [-np.pi, pitch, -np.pi]
  quat = scipyR.from_euler("xyz", rpy).as_quat()
  return np.hstack([xyz, quat, rest])

def __diffusion(state, curr_time):
  print_and_cr(f"Elapsed Time: {time.time() - state.time:.3f}s")

  diffusion_info = gen_state(state)
  state.diffusion_agent.add_obs(diffusion_info)
  with state._diffusion_lock:
    action_idx = state.diffusion_state["action_idx"]
    pred_action = state.diffusion_state["pred_action"]
    state.diffusion_state["action_idx"] = action_idx + 1
  if pred_action is None or action_idx >= len(pred_action):
    print_and_cr("WARN: No action available!")
    return state.last_diffusion_cmd, [None] * 7
  target_cmd = pred_action[action_idx]
  print_and_cr(f"Using action {action_idx+1} of {len(pred_action)}")

  if "piggy_bank" in state.config.model_name:
    diffusion_info["state"] = chopify_piggy_bank(diffusion_info["state"], True)
    target_cmd = chopify_piggy_bank(target_cmd, False)
  
  mconfig = state.config.model
  if mconfig.delta:
    clip_hi = np.array(mconfig.output_clamps)
    target_cmd = np.clip(target_cmd, -clip_hi, clip_hi)
    target_cmd += diffusion_info["state"][:len(target_cmd)]
  target_cmd[3:7] /= np.linalg.norm(target_cmd[3:7])

  target_cmd = np.clip(target_cmd, mconfig.action_low, mconfig.action_high)
  target_cmd[3:7] /= np.linalg.norm(target_cmd[3:7])

  print_and_cr(f"State  = {diffusion_info['state']}")
  print_and_cr(f"Curr   = {construct_choppose(state.arm, state.current_position)}")
  print_and_cr(f"Action = {target_cmd}")

  # Publish pose and target
  CHOPPOSE_PUBLISHER.update(state.ee_pose, state.current_position[-1])
  TARGET_CHOPPOSE_PUBLISHER.update_vector(target_cmd)
  state.tracked_objs["choppose_target"] = np.copy(target_cmd)
  pos_cmd = get_IK_from_mujoco(state.sim, state.current_position, target_vector=target_cmd)
  state.last_diffusion_cmd = pos_cmd

  return pos_cmd, [None] * 7
