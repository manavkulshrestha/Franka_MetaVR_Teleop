
import torch
from fire import Fire
import time
from polymetis import RobotInterface, GripperInterface
from src.femtomega import FemtoMega
from my_typing import Mat4x4
from util.transform import get_transform, transform_vector
from vr_franka import FrankaVR
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
from pathlib import Path
import json
from PIL import Image

HOME_Q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 60
GRIPPER_OPEN_WIDTH = 0.08


def record_data(cameras: list[FemtoMega], robot: RobotInterface, gripper: GripperInterface, joint1_offt: float,
                save_period: float, save_dir: Path, stop_recording: threading.Event) -> None:
    tp1 = time.perf_counter()
    counter = 0
    
    while not stop_recording.is_set():
        t = time.time()
        rgbs = {cam.serial_number: cam.get_rgb() for cam in cameras}
        robot_state, gripper_state = robot.get_robot_state(), gripper.get_state()
        joint_pos = robot_state.joint_positions
        joint_pos[0] -= joint1_offt

        with open(save_dir/f'{counter}.json', 'w') as f:
            json.dump({
                'timestamp': t,
                'joint_pos': joint_pos.tolist(),
                'joint_vel': robot_state.joint_velocities.tolist(),
                'gripper_width': gripper_state.width,
            }, f)
        for sn, rgb in rgbs.items():
            Image.fromarray(rgb).save(save_dir/sn/f'{counter}.png')

        tp1 += save_period
        counter += 1
        stop_recording.wait(max(0, tp1 - time.perf_counter()))


def base_T_ee(robot: RobotInterface, R_modified_R_default: R, modified_T_default: Mat4x4) -> Mat4x4:
    pos, orn = robot.get_ee_pose()
    return get_transform(
        rot_mat=(R_modified_R_default * R.from_quat(orn)).as_matrix(), 
        pos=transform_vector(modified_T_default, pos.numpy())
    )

def main(cam_serial_numbers: list[str]|None = None, *, save_dir: str, ctrl_period: float = 1/20, angled45: bool = True):
    assert not torch.cuda.is_available(), 'currently cuda not supported. ideas2 drivers are messed up'
    cam_serial_numbers = FemtoMega.connected_serial_numbers() if cam_serial_numbers is None else cam_serial_numbers
    save_dir = Path(save_dir)

    # initialize robot and gripper interfaces
    robot = RobotInterface(ip_address='localhost')
    gripper = GripperInterface(ip_address="localhost")

    # Set home position
    joint1_offt = -np.pi/4 if angled45 else 0
    HOME_Q[0] += joint1_offt
    gripper.goto(GRIPPER_OPEN_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
    robot.move_to_joint_positions(torch.tensor(HOME_Q))

    # initialize modified frame transforms
    R_default_R_modified = R.from_euler('z', joint1_offt)
    default_T_modified = get_transform(rot_mat=R_default_R_modified.as_matrix())
    R_modified_R_default = R_default_R_modified.inv()
    modified_T_default = get_transform(rot_mat=R_modified_R_default.as_matrix())

    # start recording thread
    cameras = [FemtoMega(sn) for sn in cam_serial_numbers]
    (save_dir/'robot').mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        (save_dir/cam.serial_number).mkdir(parents=True, exist_ok=True)
    stop_recording = threading.Event()
    recording_thread = threading.Thread(
        target=record_data,
        args=(cameras, robot, gripper, joint1_offt, ctrl_period, save_dir, stop_recording),
    )

    # initialize VR interface
    vri = FrankaVR(lambda: base_T_ee(robot, R_modified_R_default, modified_T_default))

    pos, orn = robot.get_ee_pose()
    robot.start_cartesian_impedance()

    first = True
    grasped = False
    try:
        while True:
            state = vri.get_robot_state()
            if state is not None:
                if first:
                    first = False
                    recording_thread.start()

                pos = torch.tensor(transform_vector(default_T_modified, state['ee_pos']))
                orn = torch.tensor((R_default_R_modified * R.from_quat(state['ee_orn'])).as_quat())

                if state["grasp"] and not grasped:
                    gripper.grasp(
                        speed=GRIPPER_SPEED,
                        force=GRIPPER_FORCE,
                        blocking=False,
                    )
                    grasped = True
                elif not state["grasp"] and grasped:
                    gripper.goto(GRIPPER_OPEN_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=False)
                    grasped = False
            robot.update_desired_ee_pose(pos, orn)

            time.sleep(ctrl_period)
    finally:
        stop_recording.set()
        recording_thread.join()
        robot.terminate_current_policy()
        for cam in cameras:
            cam.close()


if __name__ == '__main__':
    Fire(main)

# Maybe get vr stuff relative to current pose of controller