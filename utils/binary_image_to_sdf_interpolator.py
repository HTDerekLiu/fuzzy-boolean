from PIL import Image, ImageOps
from .square_grid_bilinear_interpolator import square_grid_bilinear_interpolator
import numpy as np
import scipy

def binary_image_to_sdf_interpolator(image_path, interpolator_size, val_min=-1, val_max=1):
    """
    create a bilinear interpolator to obtain sdf interpolator for a binary image

    Input:
        image_path: path to the image
        interpolator_size: resolution of the image
        val_min: (1,) min value of the grid point
        val_max: (1,) max value of the grid point

    Outputs
        interpolator: a object such that you can get interpolated data via "interpolator(pts)", pts are (m,2) query points 
    """
    image = Image.open(image_path).rotate(-90).resize((interpolator_size,interpolator_size))
    image = ImageOps.grayscale(image) 
    image = np.round(np.asarray(image)/255)
    sdf_outside = scipy.ndimage.distance_transform_edt(image)
    sdf_inside = scipy.ndimage.distance_transform_edt(1-image)
    sdf = sdf_outside - sdf_inside
    return square_grid_bilinear_interpolator(sdf, val_min, val_max)
