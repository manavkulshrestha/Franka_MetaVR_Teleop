
import torch
from fire import Fire
import time
from polymetis import RobotInterface, GripperInterface
from src.femtomega import FemtoMega
from src.my_typing import Mat4x4, Vec7
from util.transform import get_transform, transform_vector, FrameConverter
from vr_franka import FrankaVR
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
from pathlib import Path
import json
from PIL import Image
from util.io import versioned_dir
from tqdm import tqdm
from functools import partial
from typing import Callable
import multiprocessing as mp


HOME_Q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 60
GRIPPER_OPEN_WIDTH = 0.08


def save_data_mp(args):
    sample, save_dir, i = args
    with open(save_dir/f'robot/{i}.json', 'w') as f:
        json.dump(sample['robot'], f)
    for sn, rgb in sample['images'].items():
        Image.fromarray(rgb).save(save_dir/sn/f'{i}.png')

def record_data(cameras: list[FemtoMega], robot: RobotInterface, gripper: GripperInterface,
                joint1_offt: float, base_T_eef: Callable[[RobotInterface, Vec7|None], Mat4x4],
                save_period: float, save_dir: Path, stop_recording: threading.Event) -> None:
    tp1 = time.perf_counter()

    (save_dir/'robot').mkdir(parents=True, exist_ok=True)
    for cam in cameras:
        (save_dir/cam.serial_number).mkdir(parents=True, exist_ok=True)

    samples = []
    
    while not stop_recording.is_set():
        t = time.time()
        rgbs = {cam.serial_number: cam.get_rgb() for cam in cameras}
        robot_state, gripper_state = robot.get_robot_state(), gripper.get_state()
        joint_pos = np.array(robot_state.joint_positions)
        joint_pos[0] -= joint1_offt

        samples.append({
            'robot': {
                'timestamp': t,
                'joint_pos': joint_pos.tolist(),
                'joint_vel': np.array(robot_state.joint_velocities).tolist(),
                'gripper_width': gripper_state.width,
                'base_T_ee': base_T_eef(robot, joint_pos=torch.tensor(robot_state.joint_positions)).tolist(),
            },
            'images': rgbs,
        })

        # with open(save_dir/f'robot/{counter}.json', 'w') as f:
        #     json.dump({
        #         'timestamp': t,
        #         'joint_pos': joint_pos.tolist(),
        #         'joint_vel': np.array(robot_state.joint_velocities).tolist(),
        #         'gripper_width': gripper_state.width,
        #     }, f, indent=4)
        # for sn, rgb in rgbs.items():
        #     Image.fromarray(rgb).save(save_dir/sn/f'{counter}.png')

        tp1 += save_period
        stop_recording.wait(max(0, tp1 - time.perf_counter()))

    with mp.get_context("spawn").Pool(processes=mp.cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(save_data_mp, [(sample, save_dir, i) for i, sample in enumerate(samples)]),
                  desc='Saving recorded data',
                  total=len(samples)))

    # for i, sample in enumerate(tqdm(samples, desc='Saving recorded data')):
        # with open(save_dir/f'robot/{i}.json', 'w') as f:
        #     json.dump(sample['robot'], f)
        # for sn, rgb in sample['images'].items():
        #     Image.fromarray(rgb).save(save_dir/sn/f'{i}.png')


def base_T_ee(robot: RobotInterface, default_T_modified: FrameConverter, joint_pos: Vec7|None = None) -> Mat4x4:
    assert (robot is not None) or (joint_pos is not None), 'Either robot or joint_pos must be provided'
    pos, orn = robot.get_ee_pose() if joint_pos is None else robot.robot_model.forward_kinematics(joint_pos)
    
    return get_transform(
        rot_mat=default_T_modified.inv(orn.numpy(), get_rot_obj=True).as_matrix(), 
        pos=default_T_modified.inv(pos.numpy())
    )

def main(cam_serial_numbers: list[str]|None = None, *, save_dir: str, ctrl_period: float = 1/20, angled45: bool = True):
    assert not torch.cuda.is_available(), 'currently cuda not supported. ideas2 drivers are messed up'
    cam_serial_numbers = FemtoMega.connected_serial_numbers() if cam_serial_numbers is None else cam_serial_numbers
    save_dir = Path(save_dir)
    save_dir = versioned_dir(save_dir)

    # initialize robot and gripper interfaces
    robot = RobotInterface(ip_address='localhost')
    gripper = GripperInterface(ip_address="localhost")

    # Set home position
    joint1_offt = -np.pi/4 if angled45 else 0
    HOME_Q[0] += joint1_offt
    gripper.goto(GRIPPER_OPEN_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
    robot.move_to_joint_positions(torch.tensor(HOME_Q))

    # initialize modified frame transforms
    default_T_modified = FrameConverter(get_transform(rot_mat=R.from_euler('z', joint1_offt).as_matrix()))
    base_T_eef: Callable[[RobotInterface, Vec7|None], Mat4x4] = partial(base_T_ee, default_T_modified=default_T_modified)

    # start recording thread
    cameras = [FemtoMega(sn) for sn in cam_serial_numbers]
    stop_recording = threading.Event()
    recording_thread = threading.Thread(
        target=record_data,
        args=(cameras, robot, gripper, joint1_offt, base_T_eef, 1/20, save_dir, stop_recording),
    )

    # initialize VR interface
    vri = FrankaVR(partial(base_T_eef, robot=robot))

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

                pos = torch.tensor(default_T_modified(state['ee_pos']))
                orn = torch.tensor(default_T_modified(state['ee_orn']))

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
                elif state["exit"]:
                    print('Stopping tele-operation...')
                    break
            robot.update_desired_ee_pose(pos, orn)

            time.sleep(ctrl_period)
    except Exception as e:
        print(f'Error during tele-operation: {e}')
    finally:
        stop_recording.set()
        if robot.is_running_policy():
            robot.terminate_current_policy()
        print('Waiting for recording thread to finish...')
        recording_thread.join()
        print('Recording thread finished.')
        for cam in cameras:
            cam.close()


if __name__ == '__main__':
    Fire(main)

# Maybe get vr stuff relative to current pose of controller