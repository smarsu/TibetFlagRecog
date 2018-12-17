import sys
sys.path.append('/home/ffy/ffyPackage')
import os
import time

import tensorflow as tf
import numpy as np
import cv2

from tfNet import tf_net as net
import dataWash as dw


class FasterRCNN:
    def __init__(self):
        self.growth_rate_k = 12
        self.class_num = 27
        self.learning_rate = 0.1  # default is 0.1
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.batch_size = 32  #32

        self._voc = dw.DataWash("videos")

        self._net_input()

    
    def densenet_fc(self, inp, _shape, training, add_b=False):
        bn_inp = net.batch_normalization(inp, training)
        act_bn_inp = net.actv_net(bn_inp)
        fc_act_bn_inp = net.dense_net(act_bn_inp, _shape, add_b=add_b)
        return fc_act_bn_inp


    def densenet_conv(self, inp, _filter, strides, training):
        """In densenet, conv mean to BN-ReLU-Conv"""
        bn_inp = net.batch_normalization(inp, training)
        act_bn_inp = net.actv_net(bn_inp)
        conv_act_bn_inp = net.conv_net(act_bn_inp, _filter, strides)
        return conv_act_bn_inp
    

    def bottleneck_layer(self, inp, inchannel, outchannel, strides, training):
        """bottleneck: [conv 1*1, conv 3*3]
           we let each 1*1 convolution pruduce 4k feature-maps 
        """
        inp_11 = self.densenet_conv(inp, (1, 1, inchannel, 4 * self.growth_rate_k), (1, 1, 1, 1), training)
        inp_11_33 = self.densenet_conv(inp_11, (3, 3, 4 * self.growth_rate_k, outchannel), strides, training)
        return inp_11_33

    
    def dense_block(self, inp, bottleneck_num, inchannel, training):
        """Dense Block: [conv 1*1, conv 3*3] * bottleneck_num
           dense_block will concat feature map
           dense_block's conv's strides is 1
           
           For every bottleneck, it output feature map which have size of k. k is the growth rate   
        """
        precede_fmap = inp
        for _ in range(bottleneck_num):
            #inchannel = tf.shape(precede_fmap)[-1]
            fmap = self.bottleneck_layer(precede_fmap, inchannel, self.growth_rate_k, (1, 1, 1, 1), training)
            inchannel += self.growth_rate_k
            precede_fmap = tf.concat([precede_fmap, fmap], -1)
        return precede_fmap, inchannel


    def dense_net(self, bottleneck_nums=(6, 12, 24, 16)):
        """DenseNet
           bottleneck_nums = (6, 12, 64, 48) -> DenseNet-264
        """
        inp = self.images
        training = self.training

        with tf.variable_scope('DenseNet'):
            bottleneck_num_1, bottleneck_num_2, bottleneck_num_3, bottleneck_num_4 = bottleneck_nums
            logits = self.densenet_conv(inp, (7, 7, 3, 2 * self.growth_rate_k), (1, 2, 2, 1), training)
            logits = net.max_pool(logits, (1, 3, 3, 1), (1, 2, 2, 1))

            logits, inchannel = self.dense_block(logits, bottleneck_num_1, 2 * self.growth_rate_k, training)
            logits = self.densenet_conv(logits, (1, 1, inchannel, 4 * self.growth_rate_k), (1, 1, 1, 1), training)
            logits = net.avg_pool(logits, (1, 2, 2, 1), (1, 2, 2, 1))

            logits, inchannel = self.dense_block(logits, bottleneck_num_2, 4 * self.growth_rate_k, training)
            logits = self.densenet_conv(logits, (1, 1, inchannel, 4 * self.growth_rate_k), (1, 1, 1, 1), training)
            logits = net.avg_pool(logits, (1, 2, 2, 1), (1, 2, 2, 1))

            logits, inchannel = self.dense_block(logits, bottleneck_num_3, 4 * self.growth_rate_k, training)
            logits = self.densenet_conv(logits, (1, 1, inchannel, 4 * self.growth_rate_k), (1, 1, 1, 1), training)
            logits = net.avg_pool(logits, (1, 2, 2, 1), (1, 2, 2, 1))

            logits, inchannel = self.dense_block(logits, bottleneck_num_4, 4 * self.growth_rate_k, training)
            fmap_h, fmap_w = logits.get_shape()[1:3]
            #cls_logits = net.avg_pool(logits, (1, fmap_h, fmap_w, 1), (1, fmap_h, fmap_w, 1))  # for small pic, we can use avg_pool
            cls_logits = net.max_pool(logits, (1, fmap_h, fmap_w, 1), (1, fmap_h, fmap_w, 1))  # 后面得试一下不全局池化

            cls_logits = tf.reshape(cls_logits, (-1, inchannel))
            cls_logits = self.densenet_fc(cls_logits, (inchannel, self.class_num + 1), training, True)


        with tf.variable_scope('dn_box'):
            logits = net.avg_pool(logits, (1, fmap_h, fmap_w, 1), (1, fmap_h, fmap_w, 1))
            logits = tf.reshape(logits, (-1, inchannel))
            logits = self.densenet_fc(logits, (inchannel, 4), training, True)

        self.logits = logits
        self.cls_logits = cls_logits

    
    def L2_weight_decay(self, ):
        global_variables_list = tf.global_variables()
        _weight_vars = [global_variable for global_variable in global_variables_list if '_weight' in global_variable.name]

        weight_loss = 0
        for _weight in _weight_vars:
            weight_loss += self.weight_decay * tf.nn.l2_loss(_weight)
        return weight_loss


    def loss_func(self,):
        #logits = self.logits
        labels = self.labels

        obj_lb = self.obj_lb
        #obj_lgt = self.logits[:, :1]
        #obj_loss = self.obj_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=obj_lb, logits=obj_lgt))

        box_lb = self.box_lb
        box_lgt = self.logits
        box_loss = self.box_loss = tf.reduce_sum(0.5 * tf.square(box_lb - box_lgt) * obj_lb)

        #logits = self.logits[:, 5:]
        #self.classify_loss = classify_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * obj_lb)
        logits = self.cls_logits
        self.classify_loss = classify_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        self.weight_loss = weight_loss = self.L2_weight_decay()

        self.loss = (box_loss + classify_loss + weight_loss) / self.batch_size
        #self.loss = (classify_loss) / self.batch_size

        """logits = self.logits#[:, 5:]  # 这里先加上 noobj 等下去掉
        #self.classify_loss = classify_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits) * obj_lb)
        self.classify_loss = classify_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

        weight_loss = self.L2_weight_decay()

        self.loss = (classify_loss + weight_loss) / self.batch_size
        #self.loss = (obj_loss + box_loss + classify_loss + weight_loss) / self.batch_size"""


    def _optimizer(self,):
        loss = self.loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        #with tf.control_dependencies(update_ops), tf.variable_scope("DenseNet"):
        with tf.control_dependencies(update_ops), tf.variable_scope("dn_box"):
            #train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)
            self.train_op = tf.train.MomentumOptimizer(self.learning_rate, self.momentum).minimize(loss)
            #train_op = tf.train.AdamOptimizer().minimize(loss)

    
    def _net_input(self, ):
        """The values of feed_dict"""
        self.images = tf.placeholder(dtype=tf.float32, shape=(None, 256, 512, 3))
        self.labels = tf.placeholder(dtype=tf.int32, shape=(None))
        self.training = tf.placeholder(dtype=tf.bool)

        self.obj_lb = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.box_lb = tf.placeholder(dtype=tf.float32, shape=(None, 4))

    
    def _net_saver(self, ):
        """Save weight, values of batch normalize and adam value"""
        classify_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='DenseNet')
        #global_variables_list = tf.global_variables()
        #classify_vars += [global_variable for global_variable in global_variables_list if 'batch_normalization' in global_variable.name]
        self.saver = tf.train.Saver(classify_vars)

        box_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='dn_box')
        self.saver_2 = tf.train.Saver(box_vars)


    def _tf_session(self, ):
        """Start a session"""
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            try:
                #sess.run(tf.global_variables_initializer())
                #print("Init weight!")
                self.saver.restore(sess, "ckpt/DenseNet.ckpt")
                print("Restore DenseNet weight!")
                self.saver_2.restore(sess, "ckpt/dn_box.ckpt")
                print("Restore dn_box weight!")
            except:
                sess.run(tf.global_variables_initializer())
                print("Init all weight!")

            self.label2name  = self._voc.label2name 
            cnt = 0
            while True:
                self.train_classify(sess, 2, ["train_original", "train"])
                #self.value_classify(sess, "train_original")
                self.value_classify(sess, "value")
                #self.test_classify(sess, cnt, "test", "test.jpg")
                cnt += 1


    def train_classify(self, sess, epoch, model):
        for step in range(epoch):
            epoch_start_time = time.time()
            loss_value_arr = []
            for data_x, data_y, obj_lb, box_lb in self._voc.get_data(self.batch_size, model[step%2]):
            #for data_x, data_y in self._voc.get_data(self.batch_size, model[epoch%2]):
                #print(data_y[0], obj_lb[0], box_lb[0])
                _, clsf_loss, box_loss, weight_loss = sess.run([self.train_op, self.classify_loss, self.box_loss, self.weight_loss], 
                                          feed_dict={self.images: data_x,
                                                     self.labels: data_y,
                                                     self.obj_lb: obj_lb,
                                                     self.box_lb: box_lb,
                                                     self.training: True})
                
                print(clsf_loss, box_loss, weight_loss)
                loss_value_arr.append(clsf_loss + box_loss + weight_loss)

            ave_loss_value = np.mean(loss_value_arr)
            print("--step:", step, "--average loss:", ave_loss_value, "--rest time:", (time.time() - epoch_start_time) * (epoch - step - 1))
            if str(ave_loss_value) == "nan":
                exit(1)

        os.system("rm -r ckpt")
        self.saver.save(sess, "ckpt/DenseNet.ckpt")
        self.saver_2.save(sess, "ckpt/dn_box.ckpt")
        print("Save DenseNet weight!")

    
    def value_classify(self, sess, _model="value"):
        value_start_time = time.time()

        _right_count, _total_count, _total_clsf_loss, _total_box_loss, _total_obj_loss = 0, 0, 0, 0, 0
        for data_x, data_y, obj_lb, box_lb in self._voc.get_data(1, _model):
            pre_logits, clsf_loss, box_loss, obj_loss = sess.run([self.cls_logits, self.classify_loss, self.box_loss, self.weight_loss], 
                                  feed_dict={self.images: data_x,
                                             self.labels: data_y,
                                             self.obj_lb: obj_lb,
                                             self.box_lb: box_lb,
                                             self.training: False})

            _total_clsf_loss += clsf_loss
            _total_box_loss += box_loss
            _total_obj_loss += obj_loss
            
            true_label = data_y[0]  # shit!
            pre_label = np.argmax(pre_logits[0])
            #if pre_logits[0][pre_label] < 0:
                #pre_label = 27

            if true_label == pre_label:
                _right_count += 1
            else:
                pass
                #print(true_label, pre_label)
            _total_count += 1

        print("--total loss:", _total_clsf_loss, "_total_box_loss", _total_box_loss, "_total_obj_loss", _total_obj_loss,
              "--right count:", _right_count, "--total count:", _total_count, "--right rate:", _right_count / _total_count, "--total time:", time.time() - value_start_time)
        if _right_count == _total_count and "value" == _model:
            exit(0)

        
    """def value_classify(self, sess, _model="value"):
        value_start_time = time.time()

        _right_count, _total_count, _total_clsf_loss = 0, 0, 0
        for data_x, data_y in self._voc.get_data(1, _model):
            pre_logits, clsf_loss = sess.run([self.logits, self.classify_loss], 
                                  feed_dict={self.images: data_x,
                                             self.labels: data_y,
                                             self.training: False})

            _total_clsf_loss += clsf_loss
            
            true_label = data_y[0]
            pre_label = np.argmax(pre_logits[0])

            if true_label == pre_label:
                _right_count += 1
            else:
                pass
                print(true_label, pre_label)
            _total_count += 1

        print("--total loss:", _total_clsf_loss,
              "--right count:", _right_count, "--total count:", _total_count, "--right rate:", _right_count / _total_count, "--total time:", time.time() - value_start_time)
        if _right_count == _total_count and "value" == _model:
            exit(0)"""

    
    """def test_classify(self, sess, _model="test", _path="test.jpg"):
        test_start_time = time.time()

        _right_count, _total_count = 0, 0
        data_x, cvimg = self._voc.get_one_pic(_path)
        pre_logits = sess.run(self.cls_logits, 
                              feed_dict={self.images: data_x,
                                         self.training: False})
            
        pre_label = np.argmax(pre_logits)
        print("Inference:", self.label2name[pre_label], "--total time:", time.time() - test_start_time)
        cv2.imshow("test", cvimg)
        cv2.waitKey(0)"""
    def test_classify(self, sess, cnt, _model="test", _path="test.jpg"):
        test_start_time = time.time()

        _right_count, _total_count = 0, 0
        data_x, cvimg = self._voc.get_one_pic(_path)
        height_y, weight_x, channel = cvimg.shape
        pre_logits, box = sess.run([self.cls_logits, self.logits], 
                              feed_dict={self.images: data_x,
                                         self.training: False})
            
        pre_label = np.argmax(pre_logits)
        box = box[0]
        print("box", box, "Inference:", self.label2name[pre_label], "--total time:", time.time() - test_start_time)
        cvimg = cv2.rectangle(cvimg, (int(box[0] * weight_x), int(box[1] * height_y)), (int(box[2] * weight_x), int(box[3] * height_y)), (0,255,0), 4)
        cv2.imwrite(str(cnt) + "test.jpg", cvimg)
        #cv2.imshow("test", cvimg)
        #cv2.waitKey(0)




def main():
    faster_rcnn = FasterRCNN()

    faster_rcnn.dense_net()
    faster_rcnn.loss_func()
    faster_rcnn._optimizer()

    faster_rcnn._net_saver()
    faster_rcnn._tf_session()


if __name__ == "__main__":
    main()
