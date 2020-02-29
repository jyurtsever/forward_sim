import numpy as np
import cv2
import imageio
import os

"""Reads image to float values between 0 and 1 adds black to make image of shape shape"""
def imread_and_resize_to_normalized_float(im_name, shape):
    res = np.zeros(shape + (3,)).astype('uint8')
    rh = res.shape[0]
    rw = res.shape[1]
    img = cv2.imread(im_name)[...,::-1]
    if img.shape[0]/img.shape[1] > shape[0]/shape[1]: #img more vertical than psf
        img = rescale(img, height=shape[0])
        res[:, (rw - img.shape[1])//2:(rw + img.shape[1])//2, :] = img
    else: #img more horizontal than psf
        img = rescale(img, width=shape[1])
        res[(rh - img.shape[0])//2:(rh + img.shape[0])//2, :, :] = img
    return res.astype(np.float32)/255


"""Reads image to float values between 0 and 1"""
def imread_to_normalized_float(im_name):
    return cv2.imread(im_name)[...,::-1].astype(np.float32)/255


def rescale(img, width=None, height=None):
    if not width and not height:
        raise AssertionError
    if width:
        scale = width/img.shape[1]
        height = int(img.shape[0] * scale)
    else:
        scale = height / img.shape[0]
        width = int(img.shape[1] * scale)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


"""Reads image from float values between 0 and 1 with specified suffix"""
def imsave_from_normalized_float(im_name, im, suffix='.tiff'):
    save_name = os.path.split(im_name)[0] + suffix
    imageio.imsave(save_name, (im*255).astype('uint8'))