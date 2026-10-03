# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 16:22:15 2022

@author: jamyl
"""
import math
import time

import numpy as np
from numba import njit
import torch as th


DEFAULT_NUMPY_FLOAT_TYPE = np.float32

DEFAULT_TORCH_FLOAT_TYPE = th.float32
DEFAULT_TORCH_COMPLEX_TYPE = th.complex64
EPSILON_DIV = 1e-10


def getTime(currentTime, labelName, printTime=True, spaceSize=50):
	'''Print the elapsed time since currentTime. Return the new current time.'''
	if printTime:
		print(labelName, ' ' * (spaceSize - len(labelName)), ': ', round((time.perf_counter() - currentTime) * 1000, 2), 'milliseconds')
	return time.perf_counter()

def isTypeInt(array):
	'''Check if the type of a numpy array is an int type.'''
	return np.issubdtype(array.dtype, np.integer)


def getSigned(array):
	'''Return the same array, casted into a signed equivalent type.'''
	# Check if it's an unsigned dtype
	dt = array.dtype
	if dt == np.uint8:
		return array.astype(np.int16)
	if dt == np.uint16:
		return array.astype(np.int32)
	if dt == np.uint32:
		return array.astype(np.int64)
	if dt == np.uint64:
		return array.astype(np.int64)

	# Otherwise, the array is already signed, no need to cast it
	return array


@njit(inline='always')
def clamp(x, min_, max_):
    return min(max_, max(min_, x))

@njit(inline='always')
def round_half_away(x):
    """Round half away from zero (matches CUDA round() semantics)."""
    if x >= 0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

def mse(im1, im2):
    return np.linalg.norm(im1 - im2) / np.prod(im1.shape)


def divide(num, den):
    """
    Performs num = num/den

    Parameters
    ----------
    num : array[ny, nx, n_channels]

    den : array[ny, nx, n_channels]


    """
    assert num.shape == den.shape
    with np.errstate(divide='ignore', invalid='ignore'):
        num /= den

def add(A, B):
    """
    performs A += B for 2d arrays

    Parameters
    ----------
    A : array[ny, nx]

    B : array[ny, nx]


    Returns
    -------
    None.

    """
    assert A.shape == B.shape
    A += B

def timer(func, enabled, start_s=None, end_s=None, spaceSize=50):
    def wrapper(*args, **kwargs):
        t1 = time.perf_counter()
        if start_s is not None:
            print(start_s)

        out = func(*args, **kwargs)

        if end_s is not None:
            print(end_s, ' ' * (spaceSize - len(end_s)), ': ', round((time.perf_counter() - t1) * 1000, 2), 'milliseconds')

        return out
    if enabled:
        return wrapper
    else:
        return func
