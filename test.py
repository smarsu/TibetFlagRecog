import time
import os
import sys
sys.path.append("utils")

import tensorflow as tf
import numpy as np
import cv2

from utils import model
from utils import reader
from utils import get_annotations as ga

recog_times = []


def getPicPath():
    pic_path = "test.jpg" if len(sys.argv) < 2 else sys.argv[1]
    return pic_path


def test(sess, NET, cvimg, data_mode):
    start_time = time.time()
    print("data_model:", data_mode)
    res = []
    batch_image, _ = reader.getOnePic(cvimg, data_mode)
    feed_dict = {}
    feed_dict[NET.data_x] = batch_image
    feed_dict[NET.training] = False

    time1 = time.time()

    pre_lgt = sess.run(
        NET.logits,
        feed_dict = feed_dict
    )

    time2 = time.time()
    recog_time = time2 - time1
    recog_times.append(recog_time)

    pre_lgt = np.reshape(pre_lgt, [-1])
    res.append(pre_lgt)
    
    def version0(pre_lgt=pre_lgt):
        for ans in res:
            classes = [ga.idx2class[np.argmax(pre_lgt)]]
            print(classes)
            pre_lgt = np.exp(pre_lgt - max(pre_lgt)) / np.sum(np.exp(pre_lgt - max(pre_lgt)))
            idx_score = [(idx, score) for idx, score in enumerate(pre_lgt)]
            idx_score = sorted(idx_score, key=lambda x:x[-1], reverse=True)
            for idx, score in idx_score:
                print(ga.idx2class[idx], score)
            clsf_score = [(ga.idx2class[idx], score) for idx, score in idx_score]
            end_time = time.time()
            print("total_time:", end_time - start_time)
            return classes[0], clsf_score

    
    def version1():
        flag_idx = ga.class2idx['flag']
        for ans in res:
            classes = ['flag'] if ans[flag_idx] > 0 else ['background']
            print(classes, 1 / (1 + np.exp(-ans[flag_idx])))
            end_time = time.time()
            print("total_time:", end_time - start_time)
            return classes[0], [('flag', 1 / (1 + np.exp(-ans[flag_idx])))]
    
    
    return version0()


def openServer():
    NET = model.build_net()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)
    NET.saver.restore(sess, "ckpt_test/PNet.ckpt")
    print("Restore PNet weight!")

    return sess, NET


def main():
    NET = model.build_net()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        NET.saver.restore(sess, "ckpt_test/PNet.ckpt")
        print("Restore PNet weight!")

        pic_path = getPicPath()
        pic_paths = ["d.jpg"] * 102#, "test_flag.jpg", "test_gun.jpg"]
        for pic_path in pic_paths:
            print(pic_path)
            #test(sess, NET, cv2.imread(pic_path), data_mode="test")
            test(sess, NET, cv2.imread(pic_path), data_mode="train_original") 

    __recog_times = sorted(recog_times)[1:-1]
    avg_time = sum(__recog_times) / len(__recog_times)
    print('avg time:', avg_time)


if __name__ == "__main__":
    main()