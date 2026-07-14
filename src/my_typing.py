from typing import TypeAlias
from numpydantic import NDArray, Shape as S
import numpy as np

number: TypeAlias = float
Vec3: TypeAlias = NDArray[S['3'], number]
Vec4: TypeAlias = NDArray[S['4'], number]
Vec7: TypeAlias = NDArray[S['7'], number]

Mat3x3: TypeAlias = NDArray[S['3', '3'], number]
Mat4x4: TypeAlias = NDArray[S['4', '4'], number]