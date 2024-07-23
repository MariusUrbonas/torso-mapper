import nibabel as nib
import nibabel.processing as nip
import nibabel.orientations as nio
import numpy as np


def reorient_nifty(img, axcodes_to=("P", "I", "R")):
    """
    Reorients the nifti from its original orientation to another specified orientation

    Parameters:
        img: nibabel image
        axcodes_to: a tuple of 3 characters specifying the desired orientation

    Returns:
        newimg: The reoriented nibabel image
    """
    aff = img.affine
    arr = np.asanyarray(img.dataobj, dtype=img.dataobj.dtype)
    ornt_fr = nio.io_orientation(aff)
    ornt_to = nio.axcodes2ornt(axcodes_to)
    ornt_trans = nio.ornt_transform(ornt_fr, ornt_to)
    arr = nio.apply_orientation(arr, ornt_trans)
    aff_trans = nio.inv_ornt_aff(ornt_trans, arr.shape)
    newaff = np.matmul(aff, aff_trans)
    newimg = nib.Nifti1Image(arr, newaff)
    return newimg


def respace_nifty(img, voxel_spacing=(1, 1, 1), order=3):
    """Resamples the nifti from its original spacing to another specified spacing

    Params:
        img: nibabel image
        voxel_spacing: a tuple of 3 integers specifying the desired new spacing
        order: the order of interpolation

    Returns:
        new_img: The resampled nibabel image
    """
    # resample to new voxel spacing based on the current x-y-z-orientation
    aff = img.affine
    shp = img.shape
    zms = img.header.get_zooms()
    # Calculate new shape
    new_shp = tuple(
        np.rint(
            [
                shp[0] * zms[0] / voxel_spacing[0],
                shp[1] * zms[1] / voxel_spacing[1],
                shp[2] * zms[2] / voxel_spacing[2],
            ]
        ).astype(int)
    )
    new_aff = rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    return new_img


def rescale_affine(affine, shape, zooms, new_shape=None):
    """
    Taken from https://github.com/nipy/nibabel/blob/master/nibabel/affines.py due to incopatability in nibabel==5.2.1 and numpy>2.^
    """
    shape = np.asarray(shape)
    new_shape = np.array(new_shape if new_shape is not None else shape)

    s = nib.affines.voxel_sizes(affine)
    rzs_out = affine[:3, :3] * zooms / s

    # Using xyz = A @ ijk, determine translation
    centroid = nib.affines.apply_affine(affine, (shape - 1) // 2)
    t_out = centroid - rzs_out @ ((new_shape - 1) // 2)
    return nib.affines.from_matvec(rzs_out, t_out)
