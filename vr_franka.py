import time
from meta_quest_teleop.reader import MetaQuestReader
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R

from typing import Callable

from src.my_typing import Mat4x4, Vec3, Vec4


class FrankaVR:
    def __init__(self, get_base_T_ee: Callable[[], Mat4x4], base_T_vr: Mat4x4|None = None):
        self.quest_reader = MetaQuestReader()
        self.get_base_T_ee = get_base_T_ee
        self.base_T_vr = np.array([
            [0, 0, -1, 0],
            [-1, 0,  0, 0],
            [0, 1,  0, 0],
            [0, 0,  0, 1],    
        ]) if base_T_vr is None else base_T_vr

        self.prev_A = False
        self.grab_base_T_ee = None
        self.grab_base_T_eet = None

        self.controller_initialized = False

    def get_robot_state(self) -> None|dict[str, Vec3|Vec4|float|bool]:
        poses, buttons = self.quest_reader.get_transformations_and_buttons()
        if not len(poses) or not len(buttons):
            return None
        
        if not self.controller_initialized:
            self.controller_initialized = True
            print("Controller initialized. Hold A to grab the robot control")

        base_T_eet = self.base_T_vr @ poses['r']#vr_T_eet

        cur_A = buttons['A']
        if cur_A and not self.prev_A: # A was just pressed
            self.grab_base_T_ee = self.get_base_T_ee()
            self.grab_base_T_eet = base_T_eet

        state = None
        if cur_A:
            target_T = np.eye(4)
            target_T[:3, 3] = self.grab_base_T_ee[:3, 3] + (
                base_T_eet[:3, 3] - self.grab_base_T_eet[:3, 3]
            )

            R_delta = base_T_eet[:3, :3] @ self.grab_base_T_eet[:3, :3].T
            target_T[:3, :3] = R_delta @ self.grab_base_T_ee[:3, :3]

            pos = target_T[:3, 3]
            orn = R.from_matrix(target_T[:3, :3]).as_quat()
            width = (1 - buttons['rightTrig'][0]) * 0.08
            state = {
                'ee_pos': pos,
                'ee_orn': orn,
                'gripper_width': width,
                'grasp': buttons['RG'] | buttons['RTr'],
                'exit': buttons['B'],
            }
            
        self.prev_A = cur_A

        return state


def main():
    # Known issue: franka env simulation is broken for gripper opening/closing
    from src.pybullet_franka import FrankaEnv
    env = FrankaEnv()
    vri = FrankaVR(get_base_T_ee=lambda: env.base_T_ee())

    for x, y, z in [
        (0.45, -0.10, 0.04),
        (0.50,  0.05, 0.04),
        (0.55,  0.15, 0.04),
    ]:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.03, 0.03, 0.03], rgbaColor=[0, 0, 1, 1])
        p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=[x, y, z],
        )

    while True:
        time.sleep(1/60)
        state = vri.get_robot_state()
        if state is not None:
            env.move_x(state['ee_pos'], state['ee_orn'])
            env.move_gripper(state['gripper_width'])

        env.step()


if __name__ == "__main__":
    main()