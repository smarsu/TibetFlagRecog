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


def getPicPath():
    pic_path = "test.jpg" if len(sys.argv) < 2 else sys.argv[1]
    return pic_path


def image_highlight(cvimg, idx_weight, cell_size=32):
    cnt = 0
    for idx, weight in idx_weight:
        h, w = idx
        top = int(h * cell_size)
        left = int(w * cell_size)
        down = int(h * cell_size + cell_size)
        right = int(w * cell_size + cell_size)

        cvimg[top:down, left:right, :] = cvimg[top:down, left:right, :] * weight
        print(idx, weight)
        cnt += 1
    print(cnt)
    return cvimg


def test(sess, NET, cvimg, data_mode):
    print("data_model:", data_mode)

    batch_image, _ = reader.getOnePic(cvimg, data_mode)
    feed_dict = {}
    feed_dict[NET.data_x] = batch_image
    feed_dict[NET.training] = False
    f_map, dense_w, dense_b = sess.run(
        [NET.feature_map, NET.dense_w, NET.dense_b],
        feed_dict = feed_dict
    )

    print(dense_w.shape)
    print(dense_b.shape)
    print(dense_b)
    N, H, W, C = f_map.shape
    f_map = np.reshape(f_map, [N, H*W, C])
    f_map_max_idx = np.argmax(f_map, axis=1)
    print(dense_w[:, :1].tolist())
    dense_w_0 = np.reshape(dense_w[:, :1], [C])
    print(f_map.max(axis=1).tolist())
    f_map = np.reshape(f_map.max(axis=1), [C])
    print(f_map_max_idx.tolist())
    res = dense_w_0 * f_map
    print(res.tolist())
    f_map_max_idx = np.reshape(f_map_max_idx.tolist(), [C])
    idx2weight = {i * W + j:0 for i in range(H) for j in range(W)}
    for ele_product, max_idx in zip(res.tolist(), f_map_max_idx):
        idx2weight[max_idx] = idx2weight.get(max_idx, 0) + ele_product
    print(idx2weight)
    sort_idx = sorted(idx2weight.items(), key=lambda x:x[1], reverse=True)
    print(sort_idx, end='\n\n')
    sort_idx = [((idx // W, idx % W), max(0, weight / sort_idx[0][1])) for idx, weight in sort_idx]
    print(sort_idx)

    #cvimg = cv2.imread(pic_path)
    h, w, _ = cvimg.shape
    #cv2.imshow("src", cvimg)
    cvimg = image_highlight(cvimg, sort_idx, cell_size=max(h, w) / H)
    #cv2.imshow("highlight", cvimg)
    #cv2.waitKey(0)
    return cvimg


def main():
    NET = model.build_net()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        NET.saver.restore(sess, "ckpt_test/PNet.ckpt")
        print("Restore PNet weight!")

        pic_path = getPicPath()
        pic_paths = ["train_flag.jpg"]#, "test_flag.jpg", "test_gun.jpg"]
        for pic_path in pic_paths:
            print(pic_path)
            test(sess, NET, cv2.imread(pic_path), data_mode="train_original") 


if __name__ == "__main__":
    main()
