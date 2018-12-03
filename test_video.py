import time
import os
import sys
sys.path.append("utils")

import tensorflow as tf
import numpy as np
import cv2

import test
from utils import model
from utils import reader
from utils import get_annotations as ga
from utils import draw_image as di


def getPicPath():
    pic_path = "flag_pics" if len(sys.argv) < 2 else sys.argv[1]
    return pic_path


def getVideoFlow(video_path):
    """`Todo`"""
    pass


def getPicFlow(pic_path_root):
    def getCvImg(pic_path):
        '''cvimg = cv2.imread(pic_path)
        return cvimg'''
        return pic_path  # the current method is to use pic_path.


    pic_flow = map(
        getCvImg, 
        [os.path.join(pic_path_root, pic) for pic in os.listdir(pic_path_root)]
    )

    return pic_flow


def picClassify(sess, NET, pic_path, data_mode):
    class_res = test.test(sess, NET, cv2.imread(pic_path), data_mode)
    return class_res


def wtResToDir(pic_flow, dir_name, sess, NET, data_mode="test"):
    for pic_path in pic_flow:
        class_res, idx_score = picClassify(sess, NET, pic_path, data_mode)

        text = '\n'.join(
            [
                str(ga.idx2class[idx]) + ' ' + str(score) 
                for idx, score in idx_score
            ]
        )

        root_dir = os.path.join(dir_name, class_res)
        if not os.path.exists(root_dir): os.makedirs(root_dir)

        pic = os.path.split(pic_path)[-1]
        new_pic_path = os.path.join(root_dir, pic)


        cv2.imwrite(
            new_pic_path, 
            di.putText(cv2.imread(pic_path), text),
        )

    
def main():
    NET = model.build_net()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        NET.saver.restore(sess, "ckpt_test/PNet.ckpt")
        print("Restore PNet weight!")

        data_mode="train_original"
        #data_mode="test"

        pic_path_root = getPicPath()
        pic_flow = getPicFlow(pic_path_root)
        wtResToDir(pic_flow, "result", sess, NET, data_mode=data_mode)


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("total time:", t2 - t1)
