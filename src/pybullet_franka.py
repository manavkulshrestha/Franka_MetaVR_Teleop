"""
Simple franka environment in pybullet

Something specific to our setup is that we have a 45 degree offset in the base joint,
so there's an abstraction built around the move_[x|q] methods and x/q properties which handles this automatically 
and allows us to treat the robot as if it were in a standard configuration (i.e. no 45 degree offset)

Known Issue: the gripper doesn't move for the hand fixed urdf
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from util.transform import get_transform, transform_vector

from numpydantic import NDArray, Shape as S
from my_typing import Vec3, Vec4, Vec7, Mat4x4, number


class FrankaEnv:
    HOME_Q = np.array([0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4])
    FRANKA_URDF_PATH = '/home/bera/fairo/polymetis/polymetis/python/polymetis/data/fr3/fr3_franka_hand_fixed.urdf'
    GRIPPER_WIDTH = 0.08

    # def __init__(self, franka_urdf_path: str = '/home/bera/Downloads/franka_description/urdfs/fr3_franka_hand.urdf',
    def __init__(self, franka_urdf_path: str = FRANKA_URDF_PATH,
                 gravity: float = -9.81,
                 hide_collision_vis: bool = True,
                 home: bool = True,
                 angled45: bool = True,
                 headless: bool = False):
        p.connect(p.DIRECT if headless else p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity)

        self.plane = p.loadURDF('plane.urdf')
        self.robot = p.loadURDF(
            franka_urdf_path,
            useFixedBase=True,
        )

        self._joints = []
        # self._sc_joints = set()
        self._gripper_joints = []

        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[1].decode()
            joint_names = {f'fr3_joint{j}' for j in range(1, 8)}
            if name in joint_names:
                self._joints.append(i)
            # if 'sc' in p.getJointInfo(self.robot, i)[12].decode().lower():
            #     self._sc_joints.add(i)
            if 'finger_joint' in name:
                self._gripper_joints.append(i)

        self._joints = sorted(self._joints, key=lambda i: p.getJointInfo(self.robot, i)[1].decode())
        self._gripper_joints = sorted(self._gripper_joints, key=lambda i: p.getJointInfo(self.robot, i)[1].decode())

        # if hide_collision_vis:
        #     for i in self._sc_joints:
        #         p.changeVisualShape(self.robot, i, rgbaColor=[1, 1, 1, 0])

        if angled45:
            FrankaEnv.HOME_Q[0] = -np.pi/4
            self._dTm_R = R.from_euler('z', -np.pi/4)
        else:
            self._dTm_R = R.from_euler('z', 0)
        self.default_T_modified = get_transform(rot_mat=self._dTm_R.as_matrix())
        self.modified_T_default = np.linalg.inv(self.default_T_modified)
        self._mTd_R = self._dTm_R.inv()

        self.EE_LINK = find_link(self.robot, 'fr3_hand_tcp')

        if home:
            self.move_q(FrankaEnv.HOME_Q, reset=True)
            self.move_gripper(FrankaEnv.GRIPPER_WIDTH, reset=True)

    @property
    def q(self) -> Vec7:
        return np.array([p.getJointState(self.robot, i)[0] for i in self._joints], dtype=np.float64)

    def move_q(self, q: Vec7, reset=False):
        if reset:
            for i, angle in zip(self._joints, q):
                p.resetJointState(self.robot, i, angle)
        else:
            p.setJointMotorControlArray(
                self.robot,
                self._joints,
                p.POSITION_CONTROL,
                targetPositions=q,
            )

    @property
    def _x_raw(self) -> tuple[Vec3, Vec4]:
        return [np.array(e) for e in p.getLinkState(
            self.robot,
            self.EE_LINK,
            computeForwardKinematics=True
        )[:2]]

    @property
    def x(self) -> tuple[Vec3, Vec4]:
        pos, orn = self._x_raw
        pos = transform_vector(self.modified_T_default, pos)
        orn = (self._mTd_R * R.from_quat(orn)).as_quat()
        
        return pos, orn

    def move_x(self, pos: Vec3, orn: Vec4, reset: bool = False):
        pos = transform_vector(self.default_T_modified, pos)
        orn = (self._dTm_R * R.from_quat(orn)).as_quat()

        ik_sol = p.calculateInverseKinematics(
            self.robot,
            self.EE_LINK,
            targetPosition=pos,
            targetOrientation=orn,
        )

        self.move_q(ik_sol[:len(self._joints)], reset=reset)
    
    @property
    def gripper_width(self) -> float:
        return float(sum(
            p.getJointState(self.robot, joint_id)[0]
            for joint_id in self._gripper_joints
        ))
    
    def move_gripper(self, width: float, force: float = 20, reset: bool = False):
        finger_pos = width/2

        if reset:
            for i in self._gripper_joints:
                p.resetJointState(self.robot, i, finger_pos)
        else:
            p.setJointMotorControlArray(
                self.robot,
                self._gripper_joints,
            p.POSITION_CONTROL,
            targetPositions=[finger_pos] * len(self._gripper_joints),
            forces=[force] * len(self._gripper_joints),
        )
    
    def base_T_ee(self, raw: bool = False) -> Mat4x4:
        pos, orn = self._x_raw
        if raw:
            orn = R.from_quat(orn).as_matrix()
        else:
            pos = transform_vector(self.modified_T_default, pos)
            orn = (self._mTd_R * R.from_quat(orn)).as_matrix()
        
        return get_transform(rot_mat=orn, pos=pos)
    
    def step(self):
        p.stepSimulation()

    def connected(self) -> bool:
        return p.isConnected()
    
    def disconnect(self):
        p.disconnect()
        

def find_link(robot: int, name: str) -> int:
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == name:
            return i
    raise ValueError(f'{name} not found')

def create_visual_sphere(radius: float = 0.05, mass: float = 0.0, rgba_color: tuple[float, float, float, float] = (1, 0, 0, 1)) -> int:
    vis_sphere_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=rgba_color
    )
    sphere_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis_sphere_id
    )
    return sphere_id

def create_visual_cube(side_length: float = 0.05, mass: float = 0.0, rgba_color: tuple[float, float, float, float] = (1, 0, 0, 1)) -> int:
    half = side_length / 2
    vis_cube_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=(half, half, half),
        rgbaColor=rgba_color
    )
    cube_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vis_cube_id
    )
    return cube_id

# helpers for new EE safety workspace
def box_corners(lower: Vec3, upper: Vec3) -> NDArray[S['8 n, 3 xyz'], number]:  # pyright: ignore[reportInvalidTypeForm]
    x0, y0, z0 = lower
    x1, y1, z1 = upper

    return np.array([
        [x0, y0, z0],
        [x1, y0, z0],
        [x1, y1, z0],
        [x0, y1, z0],
        [x0, y0, z1],
        [x1, y0, z1],
        [x1, y1, z1],
        [x0, y1, z1],
    ])

def draw_box(corners: NDArray[S['8 n, 3 xyz'], number], color: tuple[float, float, float] = (1, 0, 0)):  # pyright: ignore[reportInvalidTypeForm]
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]

    for a, b in edges:
        p.addUserDebugLine(corners[a], corners[b], color, 2)

def axis_aligned_bbox(points: NDArray[S['8 n, 3 xyz'], number]) -> tuple[Vec3, Vec3]:  # pyright: ignore[reportInvalidTypeForm]
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return lower, upper


def main():
    env = FrankaEnv(angled45=True)

    # calculate the axis aligned bounding box of the permissive ee workspace
    safety_box = box_corners([0.1, -0.4, -0.05], [1.0, 0.4, 1.0])
    safety_box_r = env._dTm_R.apply(safety_box)
    pb_lower, pb_upper = axis_aligned_bbox(safety_box_r)
    print(f'axis aligned permissive box: lower={pb_lower}, upper={pb_upper}')
    permissive_box = box_corners(pb_lower, pb_upper)
    draw_box(permissive_box, color=(1, 0, 0))

    while True:
        env.step()
        time.sleep(1/240)


if __name__ == "__main__":
    main()