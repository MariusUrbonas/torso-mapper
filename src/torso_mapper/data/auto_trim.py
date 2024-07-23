import numpy as np


def get_max_info_bounds_along_axis(volume, axis, size=64):
    """
    Calculate the heuristic for maximum information bounds along a given axis in a volume.

    Args:
        volume (ndarray): The input volume.
        axis (int): The axis along which to calculate the bounds.
        size (int, optional): The size of the sliding window used for convolution. Defaults to 64.

    Returns:
        tuple: A tuple containing the lower and upper bounds of the maximum information region.

    """
    axis = tuple([i for i in [0, 1, 2] if i != axis])
    info_score = volume.mean(axis=axis) + volume.std(axis=axis)
    bound_low = np.argmax(np.convolve(info_score, np.ones(size), "valid"))
    bound_high = size + bound_low
    return (bound_low, bound_high)


def find_trim_bounds(volume, bounds_size=(None, 64, 64)):
    """
    Find the border of the human in a given volume using a heuristic.

    Args:
        volume (ndarray): The input volume containing the human.
        bounds_size (tuple, optional): The size of the bounds around the human.
            If a dimension is set to None, no bounds will be added in that dimension.
            Defaults to (None, 64, 64).

    Returns:
        list: A list of volume bounds representing the bounds of the human in each dimension.
    """
    volume_bounds = []
    for dim, size in enumerate(bounds_size):
        if size is None:
            volume_bounds.append(None)
            continue
        bounds = get_max_info_bounds_along_axis(volume, axis=dim, size=size)
        volume_bounds.append(bounds)
    return volume_bounds


def auto_trim_ct_scan(volume, target_size=(None, 64, 64)):
    """
    Automatically trims a CT scan volume to the specified target size such that the subject (human) is in the target box.
    Assumes that the axcodes are orriented to "P", "I", "R".

    Args:
        volume (ndarray): The input CT scan volume.
        target_size (tuple): The target size to trim the volume to. The first dimension can be None to indicate no trimming.

    Returns:
        ndarray: The trimmed CT scan volume.

    """
    bounds = find_trim_bounds(volume, target_size)
    if bounds[0] is not None:
        bound = bounds[0]
        volume = volume[bound[0] : bound[1], :, :]
    if bounds[1] is not None:
        bound = bounds[1]
        volume = volume[:, bound[0] : bound[1], :]
    if bounds[2] is not None:
        bound = bounds[2]
        volume = volume[:, :, bound[0] : bound[1]]
    return volume
