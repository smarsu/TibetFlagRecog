import math
import numpy as np
import cv2


def images_mean_std(images, _batch_num=10):
    """多张图片均值标准差"""
    def one_batch_images_mean_std(images):
        images = np.array(images, dtype=float)
        images_mean = np.mean(images)
        images_std = max(np.std(images), 1.0/np.sqrt(images[0].size))
        return images_mean, images_std

    images_mean, images_std = 0, 0
    for idx in range(_batch_num):
        batch_images = images[(len(images) // _batch_num) * idx: (len(images) // _batch_num) * (idx + 1)]
        batch_images_mean, batch_images_std = one_batch_images_mean_std(batch_images)
        images_mean += batch_images_mean
        images_std += batch_images_std
    images_mean /= _batch_num
    images_std /= _batch_num

    return images_mean, images_std


def per_image_standardization(image, mean=None, std=None):
    """单张图像标准化.
       https://www.tensorflow.org/api_docs/python/tf/image/per_image_standardization. (x - mean) / adjusted_stddev.
    """
    if mean is None:
        mean = np.mean(image)
    if std is None:
        std = max(np.std(image), 1.0/np.sqrt(image.size))

    image = image.astype(float)
    image -= mean
    image /= std
    return image


def image_rotate(image, angle=(-25.0, 25.0), center=None):
    """图像旋转并放缩."""
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)
    
    angle = np.random.uniform(low=angle[0], high=angle[1], size=None)

    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


def image_flip(img, choice_size=2):
    """图像翻转."""
    flip_target = np.random.choice(2)
    if flip_target == 0:
        img = cv2.flip(img, 1, dst=None)  # 水平镜像翻转
    return img, flip_target == 0


def image_move(img, x_axis, y_axis):
    """图像平移."""
    x = np.random.randint(*x_axis)
    y = np.random.randint(*y_axis)

    M = np.array([[1, 0, x], [0, 1, y]], np.float)  # 平移矩阵1：向x正方向平移，向y正方向平移
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return shifted


def image_rescale(img, x_scale=(0.75, 1.25), y_scale=(0.75, 1.25), xequaly=False):
    """图像x, y轴不同比例放缩."""
    '''not to use just use -> image_reshape_diff'''
    x = np.random.uniform(*x_scale)
    y = np.random.uniform(*y_scale)
    if xequaly:
        y = x

    M = np.array([[x,0,0],[0,y,0]], np.float)
    shifted = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))
    return shifted


def image_contrast_brightness(src1, contrast_scale=(0.5, 1.5), brightness_scale=(-10, 10)):  # you'd better not use contrast and light. it affect the appearance. 
    """图像亮度, 对比度调整; a调整对比度, g调整亮度."""
    a = np.random.uniform(*contrast_scale)
    g = np.random.uniform(*brightness_scale)

    h, w, ch = src1.shape

    src2 = np.zeros([h, w, ch], src1.dtype)
    dst = cv2.addWeighted(src1, a, src2, 1-a, g)
    return dst


def image_blurry(cvimg, _scale=None):
    """使图片变模糊""" 
    if _scale == None: _scale = 4
    x_scale = np.random.uniform(1, _scale)
    y_scale = x_scale  # np.random.uniform(1, 4)
    _scale = (x_scale, y_scale)

    h, w, _ = cvimg.shape
    _new_h, _new_w = math.ceil(h / _scale[0]), math.ceil(w / _scale[1])

    cvimg = cv2.resize(cvimg, (_new_w, _new_h))
    cvimg = cv2.resize(cvimg, (w, h))

    return cvimg


def image_pixels_rescale(image):
    """图像每个像素放缩"""
    image = image.astype(float)
    image /= 255
    return image


def image_rgb2gray(cvimg):
    """将 rgb 图像转换为灰度图像并且保留 shape."""
    cvimg_gray = cv2.cvtColor(cvimg, cv2.COLOR_BGR2GRAY)
    cvimg_gray = np.reshape(cvimg_gray, (cvimg_gray.shape[0], cvimg_gray.shape[1], 1))
    cvimg_gray = np.concatenate([cvimg_gray, cvimg_gray, cvimg_gray], -1)
    assert cvimg_gray.shape == cvimg.shape
    return cvimg_gray


def image_reshape_diff(cvimg, _size=None, y_diff=(-0.25, 0.25), total_diff=(-0.25, 0.25)):
    h, w, _ = cvimg.shape

    y_offset = np.random.uniform(*y_diff)
    h = (1 + y_offset) * h

    total_offset = np.random.uniform(*total_diff)
    w = (1 + total_offset) * w
    h = (1 + total_offset) * h

    '''_short, _long = _size
    if min(w, h) < _short:
        w = w / min(w, h) * _short
        h = h / min(w, h) * _short
    
    if max(w, h) > _long:
        w = w / max(w, h) * _long
        h = h / max(w, h) * _long'''

    _new_h, _new_w = math.ceil(h), math.ceil(w)

    cvimg = cv2.resize(cvimg, (_new_w, _new_h))
    return cvimg


def image_padding_square(cvimg, model="constant", pad_value=0, reshape_size=None):
    """Pad image to suqare"""
    top, bottom, left, right = 0, 0, 0, 0
    h, w, _ = cvimg.shape
    if h < w:
        bottom = w - h
    if h > w:
        right = h - w

    cvimg = cv2.copyMakeBorder(cvimg, top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
    if reshape_size is not None:
        cvimg = cv2.resize(cvimg, reshape_size)
        factor = reshape_size[0] / max(h, w)
    else:
        factor = 1

    return cvimg, factor


def image_reshape_short_side(cvimg, resized_short_side=512):
    """Used for test"""
    h, w, _ = cvimg.shape
    factor = resized_short_side / min(h, w)
    new_h, new_w = math.ceil(h * factor), math.ceil(w * factor)
    cvimg = cv2.resize(cvimg, (new_w, new_h))
    return cvimg, factor


def images_padding_uniform_size(cvimgs):
    """For a batch images, they should have same shape."""
    max_h = max([cvimg.shape[0] for cvimg in cvimgs])
    max_w = max([cvimg.shape[1] for cvimg in cvimgs])

    cvimgs = [cv2.copyMakeBorder(cvimg, 0, max_h - cvimg.shape[0], 0, max_w - cvimg.shape[1], cv2.BORDER_CONSTANT, 0) for cvimg in cvimgs]
    return cvimgs


def img_process(img, model, mean=None, std=None):
    flip_tag = 0
    if model == "train":
        img = image_blurry(img, 4)
        img = image_contrast_brightness(img, contrast_scale=(0.8, 1.2), brightness_scale=(-10, 10))
        img, flip_tag = image_flip(img)
    #img = per_image_standardization(img, mean=mean, std=std)  # Do it both for train or test image.
    return img, flip_tag


if __name__ == "__main__":
    import os 
    pic_path_arr = [os.path.join("../JPEGImages", pic) for pic in os.listdir("../JPEGImages")]
    images = [cv2.imread(pic_path) for pic_path in pic_path_arr]
    images = [image.flatten() for image in images]
    images = np.concatenate(images)
    print(images.shape)

    print(images_mean_std(images, _batch_num=10))
