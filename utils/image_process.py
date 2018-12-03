from imageProcess.data_augmentation import *


def img_process_clsf(img, model, mean=None, std=None, reshape_size=None):
    h, w, _ = img.shape
    flip_tag = False
    factor = 1
    if model == "train":
        img = image_blurry(img, 4)
        img = image_contrast_brightness(
            img, contrast_scale=(0.8, 1.2), brightness_scale=(-10, 10)
        )
        img, flip_tag = image_flip(img)
        img = image_move(img, (-w//8, w//8), (-h//8, h//8))
        img = image_reshape_diff(
            img, _size=None, y_diff=(-0.25, 0.25), total_diff=(-0.25, 0.25)
        )
        img = image_rotate(img, angle=(-15.0, 15.0))
    if "train" in model:
        img, _ = image_padding_square(
            img, model="constant", pad_value=0, reshape_size=reshape_size
        )

    img, factor = image_reshape_short_side(img, resized_short_side=224)  # 416 = 32 * 13
    img = per_image_standardization(img, mean=mean, std=std)
    return img, flip_tag, factor


def getMeanStd(img_path = '../Images'):
    import os
    import cv2
    import numpy as np
    pic_path_arr = [os.path.join("../Images", pic) for pic in os.listdir("../Images")]
    images = [cv2.imread(pic_path) for pic_path in pic_path_arr]
    images = [image.flatten() for image in images]
    images = np.concatenate(images)
    print(images.shape)

    print(images_mean_std(images, _batch_num=10))


if __name__ == "__main__":
    getMeanStd()
