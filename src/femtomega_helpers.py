import cv2
import numpy as np
from pyorbbecsdk import Pipeline, AlignFilter, PointCloudFilter, OBStreamType, OBFormat
import pyorbbecsdk as ob

def init_camera(serial_number: str):
    ctx = ob.Context()
    devices = ctx.query_devices()
    pipeline = Pipeline(devices.get_device_by_serial_number(serial_number))
    pipeline.start()
    return pipeline

def print_cameras():
    ctx = ob.Context()
    devices = ctx.query_devices()

    for i in range(len(devices)):
        dev = devices.get_device_by_index(i)
        info = dev.get_device_info()
        print(f'[{i}] name={info.get_name()}, sn={info.get_serial_number()}')

def frame_to_rgb(frame):
    assert frame.get_format() == OBFormat.MJPG
    bgr = cv2.imdecode(frame.get_data(), cv2.IMREAD_COLOR)
    return None if bgr is None else bgr[..., ::-1]

def frame_to_depth(frame, min_depth=20, max_depth=10000):
    shape = frame.get_height(), frame.get_width()
    scale = frame.get_depth_scale()

    depth = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(shape).astype(np.float32)*scale
    depth = np.where((depth > min_depth) & (depth < max_depth), depth, 0).astype(np.uint16)

    return depth

def get_rgb(pipeline):
    while True:
        frames = pipeline.wait_for_frames(100)
        if frames is None:
            continue
        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue
        return frame_to_rgb(color_frame)

def get_shaped_pcdrgb(pipeline):
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
    point_cloud_filter = PointCloudFilter()
    point_cloud_filter.set_create_point_format(OBFormat.RGB_POINT)

    while True:
        frames = pipeline.wait_for_frames(100)

        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            continue

        frame = align_filter.process(frames)
        point_cloud_frame = point_cloud_filter.process(frame)

        # rgb = frame_to_rgb(color_frame)
        # depth = frame_to_depth(depth_frame)
        # print(rgb.shape, depth.shape)

        if point_cloud_frame is None:
            continue
        return point_cloud_filter.calculate(point_cloud_frame).reshape(color_frame.get_height(), color_frame.get_width(), -1)

def shaped_pcdrgb_to_pcd_np(h_w_xyzrgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return h_w_xyzrgb[..., :3].reshape(-1, 3), h_w_xyzrgb[..., 3:].reshape(-1, 3)/255


def main():
    # import open3d as o3d
    pipeline = init_camera(0)

    # while True:
    #     try:
    #         pcdrgb = get_shaped_pcdrgb(pipeline)
    #         points, colors = shaped_pcdrgb_to_pcd_np(pcdrgb)
    #         o3d_pcd = o3d.geometry.PointCloud()
    #         o3d_pcd.points = o3d.utility.Vector3dVector(points)
    #         o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    #         o3d.visualization.draw_geometries([o3d_pcd])
    #     except KeyboardInterrupt:
    #         break

    while True:
        rgb = get_rgb(pipeline)
        cv2.imshow("RGB", rgb[..., ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    pipeline.stop()


if __name__ == "__main__":
    main()