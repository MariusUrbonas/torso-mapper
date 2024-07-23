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
    new_aff = nib.affines.rescale_affine(aff, shp, voxel_spacing, new_shp)
    new_img = nip.resample_from_to(img, (new_shp, new_aff), order=order, cval=-1024)
    return new_img
