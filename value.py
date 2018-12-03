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
from utils import draw_image as di


def value(sess, NET, data_mode="train_original", data_set="value"):
    print("data_model:", data_mode)
    start_time = time.time()

    images_collection = []

    pred_res = []
    true_res = []
    total_loss = 0
    for batch_image, batch_label in reader.getData(1, data_mode, data_set=data_set):
        feed_dict = {}
        feed_dict[NET.data_x] = batch_image
        feed_dict[NET.data_y] = batch_label
        feed_dict[NET.training] = False
        pre_lgt, loss_val = sess.run(
            [NET.logits, NET.clsf_loss],
            feed_dict = feed_dict,
        )

        total_loss += loss_val

        pre_lgt = np.reshape(pre_lgt, [-1])
        pred_res.append(pre_lgt)
        true_res += batch_label

        images_collection += batch_image
    
    def verson_0():
        right_count_dict = {}
        total_count_dict = {}
        for pred_ans, true_ans, cvimg in zip(pred_res, true_res, images_collection):
            for idx, v in enumerate(true_ans):
                if v == 1:
                    break
            total_count_dict[ga.idx2class[idx]] = \
                total_count_dict.get(ga.idx2class[idx], 0) + 1
            if pred_ans[idx] == max(pred_ans):
                right_count_dict[ga.idx2class[idx]] = \
                    right_count_dict.get(ga.idx2class[idx], 0) + 1
            # collection wrong recog
            """else:
                cvimg = di.putText(cvimg, str(pred_ans[idx]), point=(0, 30))
                cv2.imwrite("WrongRecog/" + str(pred_ans[idx]) + '.jpg', cvimg)"""
                


        for c in total_count_dict:
            print(
                c, right_count_dict.get(c, 0) / total_count_dict[c], 
                right_count_dict.get(c, 0), total_count_dict[c]
            )
        print(sum(right_count_dict.values()), sum(total_count_dict.values()))
        print("total loss:", total_loss)
    

    def verson_1():
        right_count_dict = {}
        total_count_dict = {}
        for pred_ans, true_ans in zip(pred_res, true_res):
            flag_idx = ga.class2idx["flag"]
            gun_idx = ga.class2idx["gun"]
            if true_ans[flag_idx] > 0:
                total_count_dict["flag"] = total_count_dict.get("flag", 0) + 1
                if pred_ans[flag_idx] > 0:
                    right_count_dict["flag"] = right_count_dict.get("flag", 0) + 1
            else:
                total_count_dict["background"] = total_count_dict.get("background", 0) + 1
                if pred_ans[flag_idx] <= 0:
                    right_count_dict["background"] = right_count_dict.get("background", 0) + 1
        
        for c in total_count_dict:
            print(
                c, right_count_dict.get(c, 0) / total_count_dict[c], 
                right_count_dict.get(c, 0), total_count_dict[c]
            )
        print(sum(right_count_dict.values()), sum(total_count_dict.values()))

    
    verson_0()
    end_time = time.time()
    print("total val time:", end_time - start_time)

    
def main():
    NET = model.build_net()

    #with tf.device("/cpu:0"):
        #os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
        #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        NET.saver.restore(sess, "ckpt_test/PNet.ckpt")
        #NET.saver.restore(sess, "ckpt_test/PNet.ckpt")
        print("Restore PNet weight!")

        #data_set = "value"
        data_set = "train"

        value(sess, NET, data_mode="train_original", data_set=data_set)
        value(sess, NET, data_mode="train_original", data_set="value")
        #value(sess, NET, data_mode="test", data_set=data_set) 


if __name__ == "__main__":
    main()