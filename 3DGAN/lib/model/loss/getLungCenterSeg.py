from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_erosion,  binary_closing
from skimage.filters import roberts
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import h5py
import torch
import numpy as np
from torchvision import transforms
import os


def getIMG(img,flag, num,index):
    unloader = transforms.ToPILImage()
    image = img.cpu().clone()  # clone the tensor
    # image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)

    path = '/root/project/SsLCTGAN_3DSsLD_3DPecpt_inGLungSeg/lung_img/0.3sample' + '_' + str(num) + '_' + flag + str(index) + '.jpg'
    image.save(path)


def clear_border_my(labels, buffer_size=0, bgval=0, in_place=False, mask=None):
    image = labels.cpu().clone()
    if any((buffer_size >= s for s in image.shape)) and mask is None:
        # ignore buffer_size if mask
        raise ValueError("buffer size may not be greater than image size")
    if mask is not None:
        err_msg = "image and mask should have the same shape but are {} and {}"
        assert image.shape == mask.shape, \
               err_msg.format(image.shape, mask.shape)
        if mask.dtype != bool:
            raise TypeError("mask should be of type bool.")
        borders = ~mask
    else:
        # create borders with buffer_size
        borders = np.zeros_like(image, dtype=bool)
        ext = buffer_size + 1
        slstart = slice(ext)
        slend = slice(-ext, None)
        slices = [slice(s) for s in image.shape]
        for d in range(image.ndim):
            slicedim = list(slices)
            slicedim[d] = slstart
            borders[tuple(slicedim)] = True
            slicedim[d] = slend
            borders[tuple(slicedim)] = True
    # Re-label, in case we are dealing with a binary image
    # and to get consistent labeling
    labels = label(image, background=0)
    number = np.max(labels) + 1
    # determine all objects that are connected to borders
    borders_indices = np.unique(labels[borders])
    indices = np.arange(number + 1)
    # mask all label indices that are connected to borders
    label_mask = np.in1d(indices, borders_indices)
    # create mask for pixels to clear
    mask = label_mask[labels.ravel()].reshape(labels.shape)
    if not in_place:
        image = image.clone()
    # clear border pixels
    image[mask] = bgval
    return image


# 该函数用于从给定的2D切片中分割肺
def get_segmented_lungs(im, flag,index,threshold=0.3):
    j = 0
    orign_img = im.clone()
    getIMG(orign_img,flag,index,j)
    j += 1
    # 步骤1： 二值化
    binary = im < threshold
    # binary2 = torch.where(im > threshold, 1.0, 0.0)
    # getIMG(binary2,j)
    # j += 1
    # 步骤2： 清除边界上的斑点
    cleared = clear_border_my(binary)
    # getIMG(cleared,j)
    # j += 1
    # 步骤3： 标记联通区域
    label_image = label(cleared)
    # getIMG(label_image,j)
    # j += 1
    # 保留两个最大的联通区域，即左右肺部区域，其他区域全部置为0
    areas = [r.area for r in regionprops(label_image)]
    areas.sort()
    if len(areas) > 2:
        for region in regionprops(label_image):
            if region.area < areas[-2]:
                for coordinates in region.coords:
                    label_image[coordinates[0], coordinates[1]] = 0
    binary = label_image > 0
    # 腐蚀操作，分割肺部的细节
    selem = disk(2)
    binary = binary_erosion(binary, selem)
    # 闭包操作
    selem = disk(10)
    binary = binary_closing(binary, selem)
    edges = roberts(binary)
    binary = ndi.binary_fill_holes(edges)
    # 返回最终的结果
    orign_img[~binary] = 0
    getIMG(orign_img,flag,index, j)
    j += 1
    return orign_img


if __name__ == '__main__':
    path = r"G:\LIDC90G\LIDC-HDF5-256\LIDC-IDRI-0011.20000101.3000559.1\ct_xray_data.h5"
    hdf5 = h5py.File(path, 'r')
    scan = np.asarray(hdf5['ct'])
    temp = torch.tensor(scan.copy().astype(float))
    i = 68
    tt = temp[i]
    plt.imshow(tt,cmap='gray')
    plt.show()
    mask = np.array([get_segmented_lungs(temp[i])])
    tt[~mask] = 0
    plt.imshow(tt,cmap='gray')
    plt.show()
