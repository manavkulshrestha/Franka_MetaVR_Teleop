import numpy as np
from scipy.spatial.transform import Rotation as R

from src.my_typing import Vec3, Vec4, Mat3x3, Mat4x4


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


class FrameConverter:
    def __init__(self, src_T_dst: Mat4x4):
        self.src_T_dst = src_T_dst
        self.src_R_dst = R.from_matrix(src_T_dst[:3, :3])
        self.dst_R_src = self.src_R_dst.inv()
        self.dst_T_src = np.linalg.inv(src_T_dst)

    def convert(self, pos_or_quat: Vec3|Vec4, transform: Mat4x4, rotation: R, get_rot_obj: bool = False) -> Vec3|Vec4|R:
        if len(pos_or_quat) == 3:
            return transform_vector(transform, pos_or_quat)
        elif len(pos_or_quat) == 4:
            rot = rotation * R.from_quat(pos_or_quat)
            return rot if get_rot_obj else rot.as_quat()
        else:
            raise ValueError(f'Expected 3D position or quaternion, got {pos_or_quat}')
        
    def __call__(self, pos_or_quat: Vec3|Vec4) -> Vec3|Vec4|R:
        return self.convert(pos_or_quat, self.src_T_dst, self.src_R_dst)

    def inv(self, pos_or_quat: Vec3|Vec4, get_rot_obj: bool = False) -> Vec3|Vec4|R:
        return self.convert(pos_or_quat, self.dst_T_src, self.dst_R_src, get_rot_obj=get_rot_obj)
        