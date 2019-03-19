import sys
import os
import numpy as np
import get_annotations as ga
import image_process as ip

image_train_shape = ((224, 224), )
img_mean, img_std = 138.62563120113535, 82.45444207735258


def getPicArr(Annotation_path, Image_path):
    def getPicName(f):
        return f[:-4]  # Filename contain too many '.', need [:-4] not split('.')

    annotation_pic = set(
        getPicName(f) for f in os.listdir(Annotation_path)  
    )
    image_pic = set(getPicName(f) for f in os.listdir(Image_path))
    pic_arr = annotation_pic & image_pic  # Need to guarantee own both image and annotation.
    return list(pic_arr)  # We need list to shuffle


#train_pic_arr = getPicArr(ga.Annotation_path, ga.Image_path)
#val_pic_arr = getPicArr(ga.Annotation_path_val, ga.Image_path_val)


def getPicMap(train_pic_arr):
    Annotation_path = ga.Annotation_path
    __image_train_shape = image_train_shape
    train_pic_map = {}
    for pic in train_pic_arr:
        pic_txt = os.path.join(Annotation_path, pic + '.txt')  # Clsf's file type is txt.
        data_y = ga.getAnnotation(pic_txt, ga.class2idx)
        data_y = tuple(data_y)
        train_pic_map[data_y] = train_pic_map.get(data_y, []) + [pic]
    return train_pic_map


#train_pic_map = getPicMap(train_pic_arr)


def getTrainData(batch_size, model, data_set="train"):
    def getTrainPicArr():
        max_length = max([len(__pic_arr) for __pic_arr in train_pic_map.values()])
        pic_arr = []
        for __pic_arr in train_pic_map.values():
            np.random.shuffle(__pic_arr) 
            pic_arr += __pic_arr * (max_length // len(__pic_arr))
        return pic_arr


    pic_arr = getTrainPicArr()
    Annotation_path = ga.Annotation_path
    Image_path = ga.Image_path
    __image_train_shape = image_train_shape

    np.random.shuffle(pic_arr)  
    for idx in range(len(pic_arr) // batch_size):
        for reshape_size in __image_train_shape[::-1]:
            batch_data_x = []
            batch_data_y = []
            for j in range(batch_size):
                pic_idx = idx * batch_size + j
                pic = pic_arr[pic_idx]

                pic_txt = os.path.join(Annotation_path, pic + '.txt')  # Clsf's file type is txt.
                pic_jpg = os.path.join(Image_path, pic + '.jpg')

                data_x = ga.getImage(pic_jpg)
                data_y = ga.getAnnotation(pic_txt, ga.class2idx)

                data_x, _, _ = ip.img_process_clsf(
                    data_x, model, mean=img_mean, std=img_std, 
                    reshape_size=reshape_size
                )

                batch_data_x.append(data_x)
                batch_data_y.append(data_y)
            
            batch_data_x = ip.images_padding_uniform_size(batch_data_x)

            yield batch_data_x, batch_data_y


def getData(batch_size, model, data_set="train"):
    """model: 'train' or 'train_original'"""
    if data_set == "value":
        pic_arr = val_pic_arr
        Annotation_path = ga.Annotation_path_val
        Image_path = ga.Image_path_val
        __image_train_shape = image_train_shape[:1]
    if data_set == "train":
        pic_arr = train_pic_arr
        Annotation_path = ga.Annotation_path
        Image_path = ga.Image_path
        __image_train_shape = image_train_shape

    np.random.shuffle(pic_arr)  
    for idx in range(len(pic_arr) // batch_size):
        for reshape_size in __image_train_shape[::-1]:
            batch_data_x = []
            batch_data_y = []
            for j in range(batch_size):
                pic_idx = idx * batch_size + j
                pic = pic_arr[pic_idx]

                pic_txt = os.path.join(Annotation_path, pic + '.txt')  # Clsf's file type is txt.
                pic_jpg = os.path.join(Image_path, pic + '.jpg')

                data_x = ga.getImage(pic_jpg)
                data_y = ga.getAnnotation(pic_txt, ga.class2idx)

                data_x, _, _ = ip.img_process_clsf(
                    data_x, model, mean=img_mean, std=img_std, 
                    reshape_size=reshape_size
                )

                batch_data_x.append(data_x)
                batch_data_y.append(data_y)
            
            batch_data_x = ip.images_padding_uniform_size(batch_data_x)

            yield batch_data_x, batch_data_y


def getOnePic(cvimg, model="test"):
    data_x = cvimg  # ga.getImage(path)
    data_x, _, _ = ip.img_process_clsf(
        data_x, model, mean=img_mean, std=img_std, 
        reshape_size=image_train_shape[0]
    )

    return [data_x], None


if __name__ == "__main__":
    import cv2
    #print(pic_arr)
    for batch_data_x, batch_data_y in getData(1, "train"):
        print(batch_data_x[0].shape, batch_data_y[0])
        cv2.imshow("show", batch_data_x[0])
        cv2.waitKey(0)
