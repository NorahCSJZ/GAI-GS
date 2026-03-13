import numpy as np
import math

from scene.cameras import Camera
from scipy.spatial.transform import Rotation


def generate_new_cam(r_d, tx, resolution=(360, 90), dataset_type='rfid'):
    """
    Generate camera for wireless signal rendering.

    Args:
        r_d: Rotation matrix
        tx: Translation (gateway position)
        resolution: Base resolution
        dataset_type: 'rfid' or 'ble'
    """
    rot = r_d

    if dataset_type == 'ble':
        # Camera class places camera at world_pos via: camera_center = -R @ T
        # → T must equal  -R^T @ world_pos  so that -R @ T = world_pos
        trans = tx
    else:
        # RFID: tx (r_o) is already stored in Camera T-convention
        trans = tx

    # Both RFID and BLE render the same 90×360 hemispherical image.
    # RFID: each pixel = directional power P(θ,φ) → used as spatial spectrum.
    # BLE:  omnidirectional antenna integrates over all directions →
    #       RSSI = mean(P(θ,φ)) = image[0].mean(), taken after rendering.
    fovx = np.deg2rad(180)   # tan(85°)≈11; 180° gives tan=inf → CUDA overflow
    fovy = np.deg2rad(180)
    img_width = resolution[0]
    img_height = resolution[1]

    cam = Camera(
        R=rot, colmap_id=None, T=trans,
        FoVx=fovx, FoVy=fovy,
        image=None, image_name=None, uid=None,
        invdepthmap=None, depth_params=None
    )
    cam.image_width = img_width
    cam.image_height = img_height

    return cam
