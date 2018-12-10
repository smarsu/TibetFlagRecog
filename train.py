import time
import os
import sys
sys.path.append("utils")

import tensorflow as tf
import numpy as np

from utils import model
from utils import reader
import value

epoch = 2
train_model = ["train_original", "train"]


def train(sess, epoch, model, NET):
    for step in range(epoch):
        start_time = time.time()
        total_loss = []
        cnt = 0
        for batch_image, batch_label in reader.getTrainData(NET.batch_size, model[step%len(model)]):
            feed_dict = {}
            feed_dict[NET.data_x] = batch_image
            feed_dict[NET.data_y] = batch_label
            feed_dict[NET.training] = True
            _, clsf_loss, weight_loss = sess.run(
                [NET.train_op, NET.clsf_loss, NET.weight_loss],
                feed_dict = feed_dict,
            )

            #print("clsf_loss:", clsf_loss, "weight_loss:", weight_loss)
            total_loss.append(clsf_loss)

            """if cnt % 1000 == 0:
                finish_time = time.time()
                ave_loss_value = np.mean(total_loss)
                print(
                    "--step:", step, "--average loss:", ave_loss_value, 
                )"""
            cnt += 1
        finish_time = time.time()
        ave_loss_value = np.mean(total_loss)
        print(
            "--step:", step, "--average loss:", ave_loss_value, 
            "--rest time:", (finish_time - start_time) * (epoch - step - 1)
        )
        if str(ave_loss_value) == "nan":
            exit(1)

    os.system("rm -r ckpt")
    NET.saver.save(sess, "ckpt/PNet.ckpt")
    print("Save DenseNet weight!")
        

def main():
    NET = model.build_net()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        try:
            NET.saver.restore(sess, "ckpt/PNet.ckpt")
            print("Restore PNet weight!")
        except:
            sess.run(tf.global_variables_initializer())
            print("Init all weight!")

        cnt = 0
        while True:
            train(sess, epoch, train_model, NET) 
            value.value(sess, NET, data_mode="train_original", data_set="train")
            value.value(sess, NET, data_mode="train_original")
            #value.value(sess, NET, data_mode="test")

            cnt += 1
            print("epoch: ", cnt * epoch)

if __name__ == "__main__":
    main()
