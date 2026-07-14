import numpy as np
from scipy.spatial.transform import Rotation as R


def get_transform(quat: np.ndarray = None, rot_mat: np.ndarray = None, rpy: np.ndarray = None,
                  degrees: bool = True,
                  pos: np.ndarray = None) -> np.ndarray:
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

def transform_vector(T: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ vector + T[:3, 3]