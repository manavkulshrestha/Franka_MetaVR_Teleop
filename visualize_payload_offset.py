"""
Use this script to visualize the center of mass of payload in flange frame.

You can use the GUI to move the COM in end-effector frame
The script will print out the associated COM offset in flange frame, needed for Franka to account for payload mass in impedance control.
"""

import time
import numpy as np
import pybullet as p
from fire import Fire

from src.pybullet_franka import FrankaEnv, create_visual_sphere, box_corners, axis_aligned_bbox, draw_box

from my_typing import Vec3


def find_link(robot: int, name: str) -> int:
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == name:
            return i
    raise ValueError(f'{name} not found')

def add_payload_com_gui(
        robot: int, com: tuple[float, float, float] = (0.06, 0.0, -0.113),
        flange_name: str = 'fr3_link8', ee_name: str = 'fr3_hand_tcp'
    ) -> tuple[int, int, list[int], int]:
    flange_link = find_link(robot, flange_name)
    ee_link = find_link(robot, ee_name)

    sliders = [
        p.addUserDebugParameter(f'EE COM {d}', -0.3, 0.3, com_d)
        for d, com_d in zip('xyz', com)
    ]
    marker = create_visual_sphere(radius=0.01, rgba_color=(1, 1, 0, 1))

    return flange_link, ee_link, sliders, marker


def update_payload_com_gui(
        robot: int, flange_link: int, ee_link: int,
        sliders: list, marker: int, line: int
    ) -> tuple[Vec3, Vec3, int]:
    com_ee = np.array([p.readUserDebugParameter(slider) for slider in sliders])

    flange_pos, flange_orn = p.getLinkState(robot, flange_link, computeForwardKinematics=True)[4:6]
    ee_pos, ee_orn = p.getLinkState(robot, ee_link, computeForwardKinematics=True)[4:6]
    com_world, _ = p.multiplyTransforms(ee_pos, ee_orn, com_ee, [0, 0, 0, 1])
    flange_inv_pos, flange_inv_orn = p.invertTransform(flange_pos, flange_orn)
    com_flange, _ = p.multiplyTransforms(flange_inv_pos, flange_inv_orn, com_world, [0, 0, 0, 1])

    p.resetBasePositionAndOrientation(marker, com_world, [0, 0, 0, 1])

    line = p.addUserDebugLine(
        ee_pos,
        com_world,
        [1, 1, 0],
        3,
        replaceItemUniqueId=line,
    )

    return com_ee, np.array(com_flange), line


def main(com_ee: tuple[float, float, float] = (0.06, 0.0, -0.113)):
    env = FrankaEnv()

    safety_box = box_corners([0.1, -0.4, -0.05], [1.0, 0.4, 1.0])
    safety_box_r = env._dTm_R.apply(safety_box)
    pb_lower, pb_upper = axis_aligned_bbox(safety_box_r)
    print(f'axis aligned permissive box: lower={pb_lower}, upper={pb_upper}')
    permissive_box = box_corners(pb_lower, pb_upper)
    draw_box(permissive_box, color=(1, 0, 0))

    com_flange_link, com_ee_link, com_sliders, com_marker = add_payload_com_gui(env.robot, com_ee)
    com_line = -1
    previous_com = None

    while True:
        com_ee, com_flange, com_line = update_payload_com_gui(
            env.robot,
            com_flange_link,
            com_ee_link,
            com_sliders,
            com_marker,
            com_line,
        )

        if previous_com is None or not np.allclose(com_ee, previous_com, atol=1e-4):
            print(f'com in flange frame: {com_flange.tolist()}')
            previous_com = com_ee.copy()

        env.step()
        time.sleep(1/240)
        

if __name__ == '__main__':
    main()