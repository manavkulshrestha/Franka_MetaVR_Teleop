import pybullet as p
import pybullet_data
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from transform import get_transform, transform_vector


class FrankaEnv:
    HOME_Q = np.array([0., -np.pi/4, 0., -3*np.pi/4, 0., np.pi/2, np.pi/4])

    # def __init__(self, franka_urdf_path: str = '/home/bera/Downloads/franka_description/urdfs/fr3_franka_hand.urdf',
    def __init__(self, franka_urdf_path: str = "/home/bera/fairo/polymetis/polymetis/python/polymetis/data/fr3/fr3_franka_hand_fixed.urdf",
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
        self._sc_joints = set()
        self._gripper_joints = []

        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[1].decode()
            joint_names = {f'fr3_joint{j}' for j in range(1, 8)}
            if name in joint_names:
                self._joints.append(i)
            if 'sc' in p.getJointInfo(self.robot, i)[12].decode().lower():
                self._sc_joints.add(i)
            if 'finger_joint' in name:
                self._gripper_joints.append(i)

        self._joints = sorted(self._joints, key=lambda i: p.getJointInfo(self.robot, i)[1].decode())
        self._gripper_joints = sorted(self._gripper_joints, key=lambda i: p.getJointInfo(self.robot, i)[1].decode())

        if hide_collision_vis:
            for i in self._sc_joints:
                p.changeVisualShape(self.robot, i, rgbaColor=[1, 1, 1, 0])

        self.default_T_modified = np.eye(4)
        if angled45:
            FrankaEnv.HOME_Q[0] = -np.pi/4
            self.default_T_modified = get_transform(rot_mat=R.from_euler('z', -np.pi/4).as_matrix())
        self.modified_T_default = np.linalg.inv(self.default_T_modified)

        self._dTm_R = R.from_matrix(self.default_T_modified[:3, :3])
        self._mTd_R = R.from_matrix(self.modified_T_default[:3, :3])

        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            if info[12].decode() == "fr3_hand_tcp":
                self.EE_LINK = i
                break

        if self.EE_LINK is None:
            raise RuntimeError("fr3_hand_tcp not found")

        if home:
            self.move_q(FrankaEnv.HOME_Q, reset=True)
            self.move_gripper(0.08, reset=True)

    @property
    def q(self) -> np.ndarray:
        return np.array([p.getJointState(self.robot, i)[0] for i in self._joints], dtype=np.float64)

    def move_q(self, q: np.ndarray, reset=False):
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
    def _x_raw(self) -> tuple:
        return [np.array(e) for e in p.getLinkState(
            self.robot,
            self.EE_LINK,
            computeForwardKinematics=True
        )[:2]]

    @property
    def x(self) -> np.ndarray:
        pos, orn = self._x_raw
        pos = transform_vector(self.modified_T_default, pos)
        orn = (self._mTd_R * R.from_quat(orn)).as_quat()
        
        return pos, orn

    def move_x(self, pos: np.ndarray, orn: np.ndarray, reset=False):
        pos = transform_vector(self.default_T_modified, pos)
        orn = (self._dTm_R * R.from_quat(orn)).as_quat()

        ik_sol = p.calculateInverseKinematics(
            self.robot,
            self.EE_LINK,
            targetPosition=pos,
            targetOrientation=orn,
        )

        return self.move_q(ik_sol[:len(self._joints)], reset=reset)
    
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
    
    def base_T_ee(self, raw: bool = False) -> np.ndarray:
        pos, orn = self._x_raw
        if raw:
            orn = R.from_quat(orn).as_matrix()
        else:
            pos = transform_vector(self.modified_T_default, pos)
            orn = (self._mTd_R * R.from_quat(orn)).as_matrix()
        
        return get_transform(rot_mat=orn, pos=pos)
    
    def step(self):
        p.stepSimulation()

    def connected(self):
        return p.isConnected()
    
    def disconnect(self):
        p.disconnect()
        

def create_visual_sphere(radius: float = 0.05, mass: float = 0.0, rgba_color: tuple = (1, 0, 0, 1)):
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

def create_visual_cube(side_length: float = 0.05, mass: float = 0.0, rgba_color: tuple = (1, 0, 0, 1)):
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
def box_corners(lower, upper):
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

def draw_box(corners, color=(1, 0, 0)):
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (0,4), (1,5), (2,6), (3,7),
    ]

    for a, b in edges:
        p.addUserDebugLine(corners[a], corners[b], color, 2)

def axis_aligned_bbox(points):
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    return lower, upper


# COM STUFF
def find_link(robot, name):
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[12].decode() == name:
            return i
    raise ValueError(f"{name} not found")


def add_payload_com_gui(robot, com=(0.06, 0.0, -0.113),
                        flange_name="fr3_link8",
                        ee_name="fr3_hand_tcp"):
    flange_link = find_link(robot, flange_name)
    ee_link = find_link(robot, ee_name)

    sliders = [
        p.addUserDebugParameter("EE COM x", -0.2, 0.2, com[0]),
        p.addUserDebugParameter("EE COM y", -0.2, 0.2, com[1]),
        p.addUserDebugParameter("EE COM z", -0.2, 0.3, com[2]),
    ]

    marker = create_visual_sphere(
        radius=0.01,
        rgba_color=(1, 1, 0, 1),
    )

    return flange_link, ee_link, sliders, marker


def update_payload_com_gui(robot, flange_link, ee_link, sliders, marker, line):
    com_ee = np.array([p.readUserDebugParameter(slider) for slider in sliders])

    flange_pos, flange_orn = p.getLinkState(
        robot, flange_link, computeForwardKinematics=True
    )[4:6]

    ee_pos, ee_orn = p.getLinkState(
        robot, ee_link, computeForwardKinematics=True
    )[4:6]

    com_world, _ = p.multiplyTransforms(
        ee_pos, ee_orn, com_ee, [0, 0, 0, 1]
    )

    flange_inv_pos, flange_inv_orn = p.invertTransform(
        flange_pos, flange_orn
    )

    com_flange, _ = p.multiplyTransforms(
        flange_inv_pos, flange_inv_orn,
        com_world, [0, 0, 0, 1]
    )

    p.resetBasePositionAndOrientation(marker, com_world, [0, 0, 0, 1])

    line = p.addUserDebugLine(
        ee_pos,
        com_world,
        [1, 1, 0],
        3,
        replaceItemUniqueId=line,
    )

    return com_ee, np.array(com_flange), line


def main():
    angled45 = False
    env = FrankaEnv(angled45=angled45)

    safety_box = box_corners([0.1, -0.4, -0.05], [1.0, 0.4, 1.0])
    safety_box_r = env._dTm_R.apply(safety_box)
    pb_lower, pb_upper = axis_aligned_bbox(safety_box_r)
    print(f'axis aligned permissive box: lower={pb_lower}, upper={pb_upper}')
    permissive_box = box_corners(pb_lower, pb_upper)
    draw_box(permissive_box, color=(1, 0, 0))

    com_flange_link, com_ee_link, com_sliders, com_marker = add_payload_com_gui(env.robot)
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
'''
0.0431707501411438, -0.04317067563533783, -0.009757637977600098
'''

if __name__ == "__main__":
    main()