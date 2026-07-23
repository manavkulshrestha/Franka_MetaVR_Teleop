import numpy as np
from scipy.spatial.transform import Rotation as R

from my_typing import Vec3, Vec4, Mat3x3, Mat4x4

def get_transform(quat: Vec4|None = None, rot_mat: Mat3x3|None = None, rpy: Vec3|None = None,
                  degrees: bool = True,
                  pos: Vec3|None = None) -> Mat4x4:
    assert sum(x is not None for x in [rot_mat, quat, rpy]) == 1,\
    'Exactly one of rotation matrix or quaternion or euler angles must be provided'

    T = np.eye(4)
    if rot_mat is not None:
        T[:3, :3] = rot_mat
    elif quat is not None:
        T[:3, :3] = R.from_quat(quat).as_matrix()
    elif rpy is not None:
        T[:3, :3] = R.from_euler('xyz', rpy, degrees=degrees).as_matrix()

    if pos is not None:
        T[:3, 3] = pos
    return T

def transform_vector(T: Mat4x4, vector: Vec3) -> Vec3:
    return T[:3, :3] @ vector + T[:3, 3]