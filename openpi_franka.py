from openpi_client import image_tools
from openpi_client.websocket_client_policy import WebsocketClientPolicy

import torch
from fire import Fire
import time
from polymetis import RobotInterface, GripperInterface
import numpy as np

from femtomega_helpers import init_camera, get_rgb, print_cameras
import cv2

HOME_Q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 60
GRIPPER_OPEN_WIDTH = 0.08


def build_request(*, external_rgb: np.ndarray, wrist_rgb: np.ndarray,
                  q: np.ndarray, gripper_pos: float, prompt: str,
                  angled45_correction: bool = True) -> dict:
    if angled45_correction:
        q = q+(np.pi/4, *([0]*(len(q)-1)))
    
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(external_rgb, 224, 224),
        "observation/wrist_image_left": image_tools.resize_with_pad(wrist_rgb, 224, 224),
        "observation/joint_position": q.astype(np.float32),
        "observation/gripper_position": np.asarray([1 - gripper_pos/GRIPPER_OPEN_WIDTH], dtype=np.float32),
        "prompt": prompt,
    }


def main(ctrl_period: float = 1/20, angled45: bool = True,
         model_host: str = '10.168.4.52', model_port: int = 8000, *,
         ext_cam_sn: str = 'CL25854009Y', wrs_cam_sn: str = 'CL2E453000Y', prompt: str = 'pick up the black box',
         open_loop_horizon: int = 8, qdot_scale: float = 0.15, max_qdot_rad_s: float = 0.25, gripper_threshold: float = 0.5):
    assert not torch.cuda.is_available(), 'currently cuda not supported. ideas2 drivers are messed up'
    robot = RobotInterface(ip_address='localhost')
    gripper = GripperInterface(ip_address="localhost")
    policy = WebsocketClientPolicy(model_host, model_port)

    joint1_offt = -np.pi/4 if angled45 else 0
    HOME_Q[0] = joint1_offt
    gripper.goto(GRIPPER_OPEN_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
    robot.move_to_joint_positions(torch.tensor(HOME_Q))

    external_camera, wrist_camera = init_camera(ext_cam_sn), init_camera(wrs_cam_sn)
    robot.start_joint_velocity_control(torch.zeros(len(HOME_Q)))

    action_chunk = None
    action_index = 0
    try:
        while True:
            if action_chunk is None or action_index >= min(open_loop_horizon, len(action_chunk)):
                external_rgb = get_rgb(external_camera)
                wrist_rgb = get_rgb(wrist_camera)
                request = build_request(
                    external_rgb=external_rgb,
                    wrist_rgb=wrist_rgb,
                    q=robot.get_joint_positions().numpy(),
                    gripper_pos=gripper.get_state().width,
                    prompt=prompt,
                    angled45_correction=angled45,
                )

                response = policy.infer(request)
                action_chunk = np.asarray(response['actions'], dtype=np.float32)
                if action_chunk.ndim != 2 or action_chunk.shape[1] != 8:
                    raise RuntimeError(f'Expected action chunk [horizon, 8], got {action_chunk.shape}')
                action_index = 0
                print(f'Received action chunk shape={action_chunk.shape}')

            action = action_chunk[action_index]
            action_index += 1

            qdot = np.clip(action[:7], -1.0, 1.0) * qdot_scale
            qdot = np.clip(qdot, -max_qdot_rad_s, max_qdot_rad_s)
            gripper_grasp = float(action[7]) > gripper_threshold

            print(f'dry-run qdot={np.array2string(qdot, precision=3)} grasp?={gripper_grasp}')
            # robot.update_desired_joint_velocities(torch.tensor(qdot))
            # if gripper_grasp:
            #     gripper.grasp(speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=False)
            # else:
            #     gripper.goto(GRIPPER_OPEN_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=False)

            time.sleep(ctrl_period)
    finally:
        robot.terminate_current_policy()


if __name__ == '__main__':
    Fire(main)

    # cam = init_camera('CL2E453000Y')
    # while True:
    #     rgb = get_rgb(cam)
    #     cv2.imshow("RGB", rgb[::2, ::2, ::-1])
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cam.stop()