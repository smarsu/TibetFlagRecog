import os 
import cv2

CLASSE = ['TibetFlag']
class_num = len(CLASSE)

Annotation_path = "/home/ffy/Desktop/objdataset/Annotations"
Image_path = "/home/ffy/Desktop/objdataset/Images"

Annotation_path_val = "/home/ffy/Desktop/objdataset/Annotations_val"
Image_path_val = "/home/ffy/Desktop/objdataset/Images_val"


def classToIdx(CLASSE, with_background=False):
    if with_background is True:
        CLASSE.append('background')
    
    idx2class = dict(enumerate(CLASSE))
    class2idx = {v:k for k, v in idx2class.items()}  # Dict swap keys and values. 
    return class2idx, idx2class


class2idx, idx2class = classToIdx(CLASSE, with_background = True)  # now not use softmax


def getAnnotation(path, class2idx):
    def getOnehot(annotation):
        one_hot = [0] * len(class2idx)
        for annot in annotation:
            one_hot[annot] = 1
        return one_hot


    with open(path, 'r') as f:
        annotation = [
           class2idx[c.strip()] 
           for c in f.readlines() 
           if c.strip()
        ]

    one_hot = getOnehot(annotation)
    return one_hot


def getImage(path):
    cvimg = cv2.imread(path)
    return cvimg


if __name__ == "__main__":
    print(class2idx)
    print(idx2class)
