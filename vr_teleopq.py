
import torch
from fire import Fire
import time
from polymetis import RobotInterface, GripperInterface
from src.transform import get_transform, transform_vector
from vr_franka import FrankaVR
import numpy as np
from scipy.spatial.transform import Rotation as R


HOME_Q = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 60
GRIPPER_OPEN_WIDTH = 0.08


def base_T_ee(robot, R_modified_R_default, modified_T_default):
    pos, orn = robot.get_ee_pose()
    return get_transform(
        rot_mat=(R_modified_R_default * R.from_quat(orn)).as_matrix(), 
        pos=transform_vector(modified_T_default, pos.numpy())
    )


def main(ctrl_period: float = 1/20, angled45: bool = True, gripper_cmd_tol: float = 0.002):
    assert not torch.cuda.is_available(), 'currently cuda not supported. ideas2 drivers are messed up'
    robot = RobotInterface(ip_address='localhost')
    gripper = GripperInterface(ip_address='localhost')

    joint1_offt = -np.pi/4 if angled45 else 0
    HOME_Q[0] = joint1_offt
    gripper.goto(GRIPPER_OPEN_WIDTH, speed=GRIPPER_SPEED, force=GRIPPER_FORCE, blocking=True)
    robot.move_to_joint_positions(torch.tensor(HOME_Q))

    R_default_R_modified = R.from_euler('z', joint1_offt)
    default_T_modified = get_transform(rot_mat=R_default_R_modified.as_matrix())
    R_modified_R_default = R_default_R_modified.inv()
    modified_T_default = get_transform(rot_mat=R_modified_R_default.as_matrix())

    vri = FrankaVR(lambda: base_T_ee(robot, R_modified_R_default, modified_T_default))

    q = robot.get_joint_positions()
    robot.start_joint_impedance()
    grasped = False
    try:
        while True:
            state = vri.get_robot_state()
            if state is not None:
                # ik to get target joint positions
                pos = torch.tensor(transform_vector(default_T_modified, state['ee_pos']))
                orn = torch.tensor((R_default_R_modified * R.from_quat(state['ee_orn'])).as_quat())
                q_new, valid = robot.solve_inverse_kinematics(pos, orn, q)
                if valid:
                    q = q_new

                # gripper control
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
            # joint position control
            robot.update_desired_joint_positions(q)

            time.sleep(ctrl_period)
    finally:
        robot.terminate_current_policy()


if __name__ == '__main__':
    Fire(main)

# install polymetis on 3.11
# get vr stuff relative to current pose of controller