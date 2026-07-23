
import weakref
import cv2
import numpy as np
from pyorbbecsdk import Pipeline, AlignFilter, PointCloudFilter, OBStreamType, OBFormat, Frame, Context

from numpydantic import NDArray, Shape as S
from typing import Callable, Literal, TypeAlias
from src.my_typing import number


RGBImage: TypeAlias = NDArray[S['* h, * w, 3 c'], number]
DepthImage: TypeAlias = NDArray[S['* h, * w'], number]
PCDRGBImage: TypeAlias = NDArray[S['* h, * w, 6 xyzrgb'], number]
PCD: TypeAlias = NDArray[S['* n, 3 xyz'], number]


def print_cameras():
    ctx = Context()
    devices = ctx.query_devices()
    for i in range(len(devices)):
        dev = devices.get_device_by_index(i)
        info = dev.get_device_info()
        print(f'[{i}] name={info.get_name()}, sn={info.get_serial_number()}')

def frame_to_rgb(frame: Frame) -> RGBImage|None:
    assert frame.get_format() == OBFormat.MJPG
    bgr = cv2.imdecode(frame.get_data(), cv2.IMREAD_COLOR)
    return None if bgr is None else bgr[..., ::-1]

def frame_to_depth(frame: Frame, min_depth: int = 20, max_depth: int = 10000) -> DepthImage|None:
    shape = frame.get_height(), frame.get_width()
    scale = frame.get_depth_scale()

    depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(shape).astype(np.float32)*scale
    depth = np.where((depth > min_depth) & (depth < max_depth), depth, 0).astype(np.uint16)

    return depth

def shaped_pcdrgb_to_pcd_np(h_w_xyzrgb: PCDRGBImage) -> tuple[PCD, PCD]:
    return h_w_xyzrgb[..., :3].reshape(-1, 3), h_w_xyzrgb[..., 3:].reshape(-1, 3)/255


class FemtoMega:
    def __init__(self, serial_number: str):
        self._ctx = Context()
        devices = self._ctx.query_devices()
        self.pipeline = Pipeline(devices.get_device_by_serial_number(serial_number))
        self.pipeline.start()
        self.close: Callable[[], None] = weakref.finalize(self, self.pipeline.stop)
        self.serial_number = serial_number

    @staticmethod
    def connected_serial_numbers() -> list[str]:
        ctx = Context()
        devices = ctx.query_devices()
        return [devices.get_device_by_index(i).get_device_info().get_serial_number() for i in range(len(devices))]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def get_rgb(self, wait_for_frames: int = 100) -> RGBImage:
        while True:
            frames = self.pipeline.wait_for_frames(wait_for_frames)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            if color_frame is None:
                continue
            return frame_to_rgb(color_frame)

    def get_shaped_pcdrgb(self, wait_for_frames: int = 100) -> PCDRGBImage:
        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        point_cloud_filter = PointCloudFilter()
        point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

        while True:
            frames = self.pipeline.wait_for_frames(wait_for_frames)
            if frames is None:
                continue

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if color_frame is None or depth_frame is None:
                continue

            frame = align_filter.process(frames)
            point_cloud_frame = point_cloud_filter.process(frame)
            if point_cloud_frame is None:
                continue
            return point_cloud_filter.calculate(point_cloud_frame).reshape(
                color_frame.get_height(),
                color_frame.get_width(),
                -1
            )


def main(demo: Literal['rgb', 'pcd'] = 'rgb'):
    cams = [FemtoMega(sn) for sn in FemtoMega.connected_serial_numbers()]

    match demo:
        case 'rgb':
            while True:
                for cam in cams:
                    rgb = cam.get_rgb()
                    cv2.imshow(f'{cam.serial_number}', rgb[::2, ::2, ::-1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        case 'pcd':
            import open3d as o3d
            while True:
                try:
                    pcdrgb = cams[0].get_shaped_pcdrgb()
                    points, colors = shaped_pcdrgb_to_pcd_np(pcdrgb)
                    o3d_pcd = o3d.geometry.PointCloud()
                    o3d_pcd.points = o3d.utility.Vector3dVector(points)
                    o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
                    o3d.visualization.draw_geometries([o3d_pcd])
                except KeyboardInterrupt:
                    break

    for cam in cams:
        cam.close()


if __name__ == "__main__":
    from fire import Fire
    Fire(main)