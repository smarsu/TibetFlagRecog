import sys
sys.path.append("/home/ffy/ffyPackage")
import os

from imageProcess import data_augmentation as da 
from LableCreator import box_label_translate as blt
import numpy as np
import cv2


class DataGenerate():
    def __init__(self, foreground_root, background_root):
        self.foreground_pic_arr = self.getPicArr(foreground_root)
        self.background_pic_arr = self.getPicArr(background_root)

        self.category_arr = self.getCategoryArr(foreground_root)
        self.class_num = len(self.category_arr)
        self.category2label, self.label2category = self.createCategoryToLabel()

        self.mean, self.std = 129.96301058870318, 78.39187330044088


    def getPicArr(self, path_root):
        """Get pictures, for foreground or background."""
        full_path_pic_arr = [os.path.join(root, name) for root, _, files in os.walk(path_root) for name in files]

        return full_path_pic_arr


    def getCategoryArr(self, path_root):
        #category_arr = [file_name for file_name in os.listdir(path_root)]  # get objects' name by filename, but it is disorder.
        category_arr = ['diningtable', 'boat', 'bottle', 'person', 'aeroplane',
                        'bicycle', 'car', 'cat', 'chair', 'horse', 
                        'pottedplant', 'sofa', 'motorbike', 'cow', 'train',
                         'bird', 'tvmonitor', 'dog', 'bus', 'sheep']

        return category_arr

    
    def createCategoryToLabel(self):
        """Create a subscript for each category.
           **Background will get index len(category_arr).**
        """
        category2label = {}
        label2name = {}
        for idx, category in enumerate(self.category_arr):
            category2label[category] = idx
            label2name[idx] = category
        category2label['background'] = len(self.category_arr)
        label2name[len(self.category_arr)] = 'background'

        return category2label, label2name


    def get_data(self, batch, _model):
        """Yield batch data."""
        fore_pic_arr = self.foreground_pic_arr
        back_pic_arr = self.background_pic_arr

        np.random.shuffle(fore_pic_arr)
        np.random.shuffle(back_pic_arr)

        total_iter_time = len(fore_pic_arr) // batch  # 重点在于走完 foreground

        for idx in range(total_iter_time):
            batch_image, batch_label = [], []
            for j in range(batch):
                fore_idx = j + idx * batch
                back_idx = (j + idx * batch) % len(back_pic_arr)

                fore_img = self.getCVImg(fore_pic_arr[fore_idx])
                back_img = self.getCVImg(back_pic_arr[back_idx])
                ctgr = self.getCategoryIdxFromPath(fore_pic_arr[fore_idx])

                cmb_img, box = self.combinateImg(fore_img, back_img)
                cmb_img, flip_tag = da.img_process(cmb_img, _model, mean=self.mean, std=self.std)  # da

                if flip_tag:
                    h, w, _ = cmb_img.shape
                    x1, y1, x2, y2 = box
                    box = (w - x2, y1, w - x1, y2)

                bboxs_ctgrs = [(box, ctgr)]

                label = blt.bboxToSSDLable(bboxs_ctgrs, self.class_num)

                batch_image.append(cmb_img)
                batch_label.append(label)

            yield batch_image, batch_label

    
    def get_one_pic(self, _path):
        """Test"""
        cvimg = cv2.imread(_path)
        dup_cvimg = cvimg.copy()
        cvimg, factor = da.image_padding_square(cvimg)
        cvimg, flip_tag = da.img_process(cvimg, "test", mean=self.mean, std=self.std)
        
        assert 0 == flip_tag

        return [cvimg], dup_cvimg

    
    def getCVImg(self, img_path):
        cvimg = cv2.imread(img_path)
        return cvimg

    
    def getCategoryIdxFromPath(self, path):
        category = path.split('/')[-2]  # for certain
        category_idx = self.category2label[category]
        return category_idx

    
    def combinateImg(self, fore_img, back_img, img_shape=(512, 512), fore_img_arr=None):
        """Put one fore image on background
           **Todo** list of fore images.
        """
        def processForeImg(fore_img):
            """DA for fore images.
               Key process:
                1. multi_scale
                2. flip
            """
            cvimg = fore_img

            cvimg, _ = da.image_flip(cvimg)
            cvimg = da.image_reshape_diff(cvimg)

            return cvimg

        
        def reshapeBackImg(back_img):
            """Reshape back img for better train."""
            back_img = cv2.resize(back_img, img_shape)
            return back_img

        
        def putForeImgOnBackImg(fore_img, back_img):
            """Key process:
                1. multi_location
            """
            fore_height, fore_weight, _ = fore_img.shape
            back_height, back_weight, _ = back_img.shape
            
            x1 = np.random.randint(0, (back_weight - fore_weight))
            y1 = np.random.randint(0, (back_height - fore_height))
            x2 = x1 + fore_weight
            y2 = y1 + fore_height

            back_img[y1:y2,x1:x2,:] = fore_img
            box = (x1, y1, x2, y2)
            return back_img, box
        
        proed_fore_img = processForeImg(fore_img)
        reshape_back_img = reshapeBackImg(back_img)
        cmb_img, box = putForeImgOnBackImg(proed_fore_img, reshape_back_img)
        return cmb_img, box

    
    def bboxToSSDLable(self, bbox, ctgr, receptive_field_pyramid=None):
        """Generate ssd labels."""
        receptive_field_pyramid = (32, 64, 128, 256, 512)  # the default receptive field pyramid

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        short = min(w, h)
        for idx, receptive_field in enumerate(receptive_field_pyramid):
            if short - receptive_field < receptive_field // 2:
                chosen_receptive_field_idx = idx
                break

        res = []
        for idx, receptive_field in enumerate(receptive_field_pyramid):
            chosed_bbox = None if not idx == chosen_receptive_field_idx else bbox
            obj, category, boxReg = self.bboxToYoloLable(chosed_bbox, ctgr, receptive_field)
            res.append((obj, category, boxReg))

        return res
            
        
    def bboxToYoloLable(self, bbox, ctgr, stripe, img_shape=(512, 512)):
        """Generate yolo labels.
           Used for single box.
        """
        img_w, img_h = img_shape
        w_cell_num, h_cell_num = img_w // stripe, img_h // stripe

        obj = [[0 for _ in range(w_cell_num)] for _ in range(h_cell_num)]
        category = [[[0 for _ in range(self.class_num)] for _ in range(w_cell_num)] for _ in range(h_cell_num)]
        boxReg = [[[0, 0, 0, 0] for _ in range(w_cell_num)] for _ in range(h_cell_num)]

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            true_height = y2 - y1
            true_width = x2 - x1

            x_cell_idx = int(mid_x / stripe)
            y_cell_idx = int(mid_y / stripe)

            obj[y_cell_idx][x_cell_idx] = 1
            category[y_cell_idx][x_cell_idx] = self.label2onehot(ctgr)

            cell_x = stripe * x_cell_idx + stripe / 2
            cell_y = stripe * y_cell_idx + stripe / 2
            cell_w = stripe
            cell_h = stripe

            t_x = (mid_x - cell_x) / cell_w
            t_y = (mid_y - cell_y) / cell_h
            t_w = np.log(true_width / cell_w)
            t_h = np.log(true_height / cell_h)

            boxReg[y_cell_idx][x_cell_idx] = [t_x, t_y, t_w, t_h]

        return obj, category, boxReg

    
    def label2onehot(self, label):
        onehot = [0] * label + [1] + [0] * (self.class_num - 1 - label)
        onehot = onehot[:self.class_num]  # for background
        return onehot


def main():
    voc_2012 = DataGenerate("/home/ffy/Desktop/tfstudy/VOC/foreground", "/home/ffy/Desktop/tfstudy/VOC/background")
    for batch_image, batch_label in voc_2012.get_data(1, "train"):
        cvimg = batch_image[0]
        print(len(batch_label[0]))
        for i in range(len(batch_label[0])):
            cvimg = blt.putResOnImg(cvimg, batch_label[0][i], 32 * 2**i, voc_2012.label2category)
            print(32 * 2**i)
        cv2.imshow("test", cvimg)
        cv2.waitKey(0)


if __name__ == "__main__":
    main()
