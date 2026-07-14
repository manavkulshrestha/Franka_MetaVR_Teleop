import cv2
from fire import Fire
from src.femtomega_helpers import get_rgb, init_camera
from util.io import yaml_to_dict
from typing import Literal


def main(cam_type: Literal['ext', 'wrs'] = 'wrs'):
    args = yaml_to_dict('args.yaml')
    cam = init_camera(args[f'{cam_type}_cam_sn'])
    while True:
        rgb = get_rgb(cam)
        cv2.imshow("RGB", rgb[::2, ::2, ::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.stop()


if __name__ == '__main__':
    Fire(main)