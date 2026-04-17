import numpy as np

from scene.cameras import Camera


def generate_new_cam(r_d, tx, resolution=(360, 90)):
    """Generate a receiver-centric camera for RFID signal rendering."""
    rot = r_d
    trans = tx

    fovx = np.deg2rad(180)
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
