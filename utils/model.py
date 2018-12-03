import tensorflow as tf
import get_annotations as ga
from tfNet import tf_net
from tfNet import tf_loss
from CNNFramework import ResNet


class model:
    def __init__(self):
        self.batch_size = 32  # 16, but it is said that 32 is a little fine
        self.weight_decay = 1e-4
        '''
            at first, set to 1 or 0.1, it can't converge rightly.
            so we reduce it to 0.001, it can converge slowly, but the accurate of the test samples is low and unstable
            when i change it to 0.0001, the loss of training data can get down clearly, at the the test samples' accurate grow up stably(the accuracy will not wave havily with the iter of training)
        '''
        self.learning_rate = 0.001  # if big, it will close to one class unreasonbly.  0.0001 is fine
        self.momentum = 0.9
        self.class_num = ga.class_num
        self.label_num = self.class_num + 1  # if not background


    def net_input(self):
        self.data_x = tf.placeholder(dtype=tf.float32)
        self.data_y = tf.placeholder(dtype=tf.float32)
        self.training = tf.placeholder(dtype=tf.bool)

    
    def net(self):
        """If use DenseNet, remember to use `densenet_conv()`"""
        data_flow = self.data_x
        training = self.training

        data_flow, channel = ResNet.res_net(data_flow, training)
        self.feature_map = data_flow
        with tf.variable_scope("Net"):
            #print("fmap_shape:", data_flow.get_shape().as_list())  # `fmap_shape: [None, None, None, 512]`
            #data_flow = tf_net.global_avg_pool(data_flow, channel)
            data_flow = tf_net.global_max_pool(data_flow, channel)
            """data_flow = tf_net.conv_net(
                data_flow, 
                (1, 1, channel, self.label_num), 
                (1, 1, 1, 1), 
                padding="SAME", 
                add_b=True
            )"""

            data_flow = tf.reshape(data_flow, [-1, channel])
            data_flow, dense_w, dense_b = tf_net.dense_net(
                data_flow, 
                (channel, self.label_num), 
                add_b=True,
                ret_wb = True
            )

            self.dense_w = dense_w
            self.dense_b = dense_b
        
        self.logits = data_flow

    
    def loss_func(self):  
        logits = self.logits
        labels = self.data_y

        logits = tf.reshape(logits, [-1, self.label_num])
        clsf_loss = self.clsf_loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=logits, labels=labels
            )
        ) / self.batch_size
        
        weight_loss = self.weight_loss \
            = tf_loss.L2_weight_decay(self.weight_decay)
        
        self.loss = clsf_loss + weight_loss

    
    def optimizer(self):
        loss = self.loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops), tf.variable_scope("Opt"):
            self.train_op = tf.train.MomentumOptimizer(
                self.learning_rate, self.momentum
            ).minimize(loss)

    
    def net_saver(self):
        self.saver = tf.train.Saver()


def build_net():
    NET = model()
    NET.net_input()
    NET.net()
    NET.loss_func()
    NET.optimizer()
    NET.net_saver()
    return NET


if __name__ == "__main__":
    import os
    import numpy as np
    import random
    import math

    NET = build_net()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        while True:
            h = random.randint(32, 2000)
            w = random.randint(32, 2000)

            np_image = np.zeros([1, h, w, 3])
            
            feed_dict = {}
            feed_dict[NET.data_x] = np_image
            feed_dict[NET.training] = True

            pre_lgt = sess.run(NET.logits, feed_dict=feed_dict)

            def cal_fmap(in_size):
                for _ in range(5):
                    ret_size = math.ceil(in_size / 2)
                    in_size = ret_size
                return ret_size

            guess_h = cal_fmap(h)
            guees_w = cal_fmap(w)

            assert pre_lgt.shape[1:3] == (guess_h, guees_w), \
                "pre_lgt_shape: %s\tguess_shape: %s" \
                % (pre_lgt.shape[1:3], (guess_h, guees_w))
            print(pre_lgt.shape[1:3], (guess_h, guees_w), (h, w))
