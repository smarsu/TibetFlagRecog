import time
import os

import tensorflow as tf
import numpy as np

import read_data


def conv_net(inp, filter, strides, padding="SAME", add_b=False):
    """create a conv layer"""
    conv_w = tf.Variable(tf.truncated_normal(shape=filter, stddev=0.01))
    conv_h = tf.nn.conv2d(inp, conv_w, strides, padding)
    if add_b:
        conv_b = tf.Variable(tf.truncated_normal(shape=[filter[-1]], stddev=0.01))
        conv_h += conv_b
    return conv_h


def dense_net(inp, shape, add_b=False):
    """create a dense layer"""
    dense_w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.001))
    dense_h = tf.matmul(inp, dense_w)
    if add_b:
        dense_b = tf.Variable(tf.truncated_normal(shape=[shape[-1]], stddev=0.001))
        dense_h += dense_b
    return dense_h


def batch_normalization(inp, training):
    """batch_normalize for next layer's input"""
    inp_norm = tf.layers.batch_normalization(inp, training=training)
    return inp_norm


def actv_net(inp, model="leaky_relu"):
    """actvation function"""
    if model == "leaky_relu":
        inp_actv = tf.maximum(0.1 * inp, inp)
    return inp_actv


def max_pool(inp, w_shape, strides, padding="SAME"):
    pool_h = tf.nn.max_pool(inp, w_shape, strides, padding)
    return pool_h


def avg_pool(inp, w_shape, strides, padding="SAME"):
    pool_h = tf.nn.avg_pool(inp, w_shape, strides, padding)
    return pool_h


def drop_out(inp, keep_prob):
    """drop out layer"""
    drop_h = tf.nn.dropout(inp, keep_prob)
    return drop_h


def identity_shortcut_conv(inp, channel_in, channel_out, training):
    """H(x) = F(x) + x"""
    assert channel_in == channel_out
    x_shortcut = inp

    conv_h1 = conv_net(inp, (3, 3, channel_in, channel_out), (1, 1, 1, 1))
    conv_h1 = batch_normalization(conv_h1, training)
    conv_h1 = actv_net(conv_h1)

    conv_h2 = conv_net(conv_h1, (3, 3, channel_out, channel_out), (1, 1, 1, 1))
    conv_h2 = batch_normalization(conv_h2, training)

    add_result = tf.add(x_shortcut, conv_h2)
    add_result = actv_net(add_result)
    return add_result


def match_dimen_shortcut_conv(inp, channel_in, channel_out, training, strides=2):
    """H(x) = F(x) + Wx"""
    #assert channel_out == 2 * channel_in or channel_in == 2 * channel_out
    x_shortcut = inp

    conv_h1 = conv_net(inp, (3, 3, channel_in, channel_out), (1, strides, strides, 1))
    conv_h1 = batch_normalization(conv_h1, training)
    conv_h1 = actv_net(conv_h1)

    conv_h2 = conv_net(conv_h1, (3, 3, channel_out, channel_out), (1, 1, 1, 1))
    conv_h2 = batch_normalization(conv_h2, training)

    x_shortcut = conv_net(x_shortcut, (1, 1, channel_in, channel_out), (1, strides, strides, 1))
    x_shortcut = batch_normalization(x_shortcut, training)  # is there a batch_normalization?

    add_result = tf.add(x_shortcut, conv_h2)
    add_result = actv_net(add_result)
    return add_result


def identity_bottleneck_conv(inp, channel_in, channel_out, training):
    x_shortcut = inp
    channel_out1, channel_out2, channel_out3 = channel_out

    conv_h1 = conv_net(inp, (1, 1, channel_in, channel_out1), (1, 1, 1, 1))
    conv_h1 = batch_normalization(conv_h1, training)
    conv_h1 = actv_net(conv_h1)

    conv_h2 = conv_net(conv_h1, (3, 3, channel_out1, channel_out2), (1, 1, 1, 1))
    conv_h2 = batch_normalization(conv_h2, training)
    conv_h2 = actv_net(conv_h2)

    conv_h3 = conv_net(conv_h2, (1, 1, channel_out2, channel_out3), (1, 1, 1, 1))
    conv_h3 = batch_normalization(conv_h3, training)

    add_result = tf.add(x_shortcut, conv_h3)
    add_result = actv_net(add_result)
    return add_result


def match_dimen_bottleneck_conv(inp, channel_in, channel_out, training, strides=2):
    x_shortcut = inp
    channel_out1, channel_out2, channel_out3 = channel_out

    conv_h1 = conv_net(inp, (1, 1, channel_in, channel_out1), (1, strides, strides, 1))
    conv_h1 = batch_normalization(conv_h1, training)
    conv_h1 = actv_net(conv_h1)

    conv_h2 = conv_net(conv_h1, (3, 3, channel_out1, channel_out2), (1, 1, 1, 1))
    conv_h2 = batch_normalization(conv_h2, training)
    conv_h2 = actv_net(conv_h2)

    conv_h3 = conv_net(conv_h2, (1, 1, channel_out2, channel_out3), (1, 1, 1, 1))
    conv_h3 = batch_normalization(conv_h3, training)

    x_shortcut = conv_net(x_shortcut, (1, 1, channel_in, channel_out3), (1, strides, strides, 1))
    x_shortcut = batch_normalization(x_shortcut, training)  # is there a batch_normalization?

    add_result = tf.add(x_shortcut, conv_h3)
    add_result = actv_net(add_result)
    return add_result


def res_net50(data_x, keep_prob, training):
    with tf.name_scope("resnet_classify"):
        data_logit = conv_net(data_x, (7, 7, 3, 64), (1, 2, 2, 1))  # stride: 2
        data_logit = batch_normalization(data_logit, training)
        data_logit = actv_net(data_logit)

        data_logit = max_pool(data_logit, (1, 3, 3, 1), (1, 2, 2, 1))  # stride: 2

        data_logit = match_dimen_bottleneck_conv(data_logit, 64, (64, 64, 256), training, strides=1)
        for _ in range(2):
            data_logit = identity_bottleneck_conv(data_logit, 256, (64, 64, 256), training)

        data_logit = match_dimen_bottleneck_conv(data_logit, 256, (128, 128, 512), training)  # stride: 2
        for _ in range(3):
            data_logit = identity_bottleneck_conv(data_logit, 512, (128, 128, 512), training)

        data_logit = match_dimen_bottleneck_conv(data_logit, 512, (256, 256, 1024), training)  # stride: 2
        for _ in range(5):
            data_logit = identity_bottleneck_conv(data_logit, 1024, (256, 256, 1024), training)

        data_logit = match_dimen_bottleneck_conv(data_logit, 1024, (512, 512, 2048), training)  # stride: 2
        data_logit = identity_bottleneck_conv(data_logit, 2048, (512, 512, 2048), training)
        data_logit_classify = data_logit = identity_bottleneck_conv(data_logit, 2048, (512, 512, 2048), training)

        data_logit = avg_pool(data_logit, (1, 2, 2, 1), (1, 2, 2, 1))
        data_logit = tf.reshape(data_logit, [-1, 2048])

        data_logit = dense_net(data_logit, [2048, 190], add_b=True)  # remember change for value

    """with tf.name_scope("resnet_regress"):
        data_logit_classify = match_dimen_shortcut_conv(data_logit_classify, 512, 300, training, strides=1)
        data_logit_classify = conv_net(data_logit_classify, (1, 1, 300, 300), (1, 1, 1, 1), add_b=True)"""

    return data_logit


def res_net(data_x, keep_prob, training):
    """create a resnet network

    Returns:
        data_x: tf.placeholder().  Input of the whole network. A 4-D shape [batch_size, height, width, channel]
        training: tf.placeholder().  True/False. Used for batch_normalization
        data_logit: Tensor.  The predicted output
    """
    with tf.name_scope("resnet_classify"):
        data_logit = conv_net(data_x, (7, 7, 3, 64), (1, 2, 2, 1))  # stride: 2
        data_logit = batch_normalization(data_logit, training)
        data_logit = actv_net(data_logit)

        data_logit = max_pool(data_logit, (1, 3, 3, 1), (1, 2, 2, 1))  # stride: 2

        data_logit = identity_shortcut_conv(data_logit, 64, 64, training)
        data_logit = identity_shortcut_conv(data_logit, 64, 64, training)
        data_logit = identity_shortcut_conv(data_logit, 64, 64, training)

        data_logit = match_dimen_shortcut_conv(data_logit, 64, 128, training)  # stride: 2
        data_logit = identity_shortcut_conv(data_logit, 128, 128, training)
        data_logit = identity_shortcut_conv(data_logit, 128, 128, training)
        data_logit = identity_shortcut_conv(data_logit, 128, 128, training)

        data_logit = match_dimen_shortcut_conv(data_logit, 128, 256, training)  # stride: 2
        data_logit = identity_shortcut_conv(data_logit, 256, 256, training)
        data_logit = identity_shortcut_conv(data_logit, 256, 256, training)
        data_logit = identity_shortcut_conv(data_logit, 256, 256, training)
        data_logit = identity_shortcut_conv(data_logit, 256, 256, training)
        data_logit = identity_shortcut_conv(data_logit, 256, 256, training)

        data_logit = match_dimen_shortcut_conv(data_logit, 256, 512, training)  # stride: 2
        data_logit = identity_shortcut_conv(data_logit, 512, 512, training)
        data_logit = identity_shortcut_conv(data_logit, 512, 512, training)

        data_logit = avg_pool(data_logit, (1, 2, 2, 1), (1, 2, 2, 1))
        data_logit = tf.reshape(data_logit, [-1, 512])

        data_logit_embed = dense_net(data_logit, [512, 300], add_b=True)  # remember change for value
        data_logit_class = dense_net(data_logit, [512, 157], add_b=True)  # remember change for value
    return data_logit_embed


def mlp_net(data_y, keep_prob, training):
    """Multi-Layer Perceptron used to map embed to 512 dimen"""
    with tf.name_scope("resnet_regress"):
        data_logit = data_y

    return data_logit


def x_attention_y(data_x_logit, data_y_logit):
    """x pay different attention to y"""
    data_x_logit = tf.reshape(data_x_logit, [-1, 4, 300])
    data_y_logit = tf.reshape(data_y_logit, [-1, 1, 300])

    data_score = tf.nn.softmax(tf.reduce_sum(data_x_logit * data_y_logit, -1))  # shape(-1, 4)
    data_score = tf.reshape(data_score, [-1, 4, 1])

    weight_x_logit = data_x_logit * data_score
    weight_sum_x_logit = tf.reduce_sum(weight_x_logit, -2)  # shape(-1, 300)

    data_y_logit = tf.reshape(data_y_logit, [-1, 300])

    return weight_sum_x_logit, data_y_logit


def smoothL1(logtis, labels):
    """A stronger loss func for sum_square"""
    diff = logtis - labels

    def less_loss(inp):
        oup = 0.5 * inp * inp
        return oup

    
    def great_loss(inp):
        oup = tf.abs(inp) - 0.5
        return oup


    smooth_loss = tf.where(tf.greater(tf.abs(diff), 1.0), great_loss(diff), less_loss(diff))
    return smooth_loss


def dot_loss(logits, labels):
    """calculate sim between two vector. minmum -sim"""

    # similar 为相似度越大越好
    # loss 为损失越小越好 -> loss = -similar
    similar = tf.reduce_sum(logits * labels, -1)
    softmax_logits = tf.nn.softmax(similar)
    
    return softmax_logits


def att_loss(logits, user_att, data_y, logits_2):
    dis_logits = tf.square(logits - user_att)
    dis_logits_2 = tf.square(logits_2 - user_att)

    user_att_loss = tf.reduce_sum(tf.maximum(1 + dis_logits - dis_logits_2, 0))

    user_att_loss /= batch_size
    return user_att_loss, user_att_loss


def cls_loss(logits, labels):
    classify_loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    classify_loss /= batch_size
    return classify_loss, logits

def loss_func(data_y, logits, data_x2_logit, user_att, batch_size):
    """calculate classify loss"""
    loss, cos_similar = att_loss(logits, user_att, data_y, data_x2_logit)

    #loss, cos_similar = cls_loss(logits, data_y)

    """for train, we need loss and optimizer"""
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops), tf.name_scope("resnet_classify"):
        train_op = tf.train.GradientDescentOptimizer(1).minimize(loss)
        #train_op = tf.train.AdamOptimizer().minimize(loss)
    return loss, train_op, cos_similar


def tf_sess(data_x, data_x2, data_y, user_attr, keep_prob, training, loss, train_op, data_logit, batch_size, model="train"):
    """
        tf session for train or test. 
        First restore or init weight.

        train: optimize loss.
        test: get data_predict.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # saver for resnet_classify, ...
        resnet_classify_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_classify')
        global_variables_list = tf.global_variables()
        resnet_classify_vars += [global_variable for global_variable in global_variables_list if 'batch_normalization_1' in global_variable.name
                                                                                              or 'batch_normalization/' in global_variable.name
                                                                                              or 'batch_normalization_2' in global_variable.name
                                                                                              or 'batch_normalization_3' in global_variable.name
                                                                                              or 'batch_normalization_4' in global_variable.name
                                                                                              or 'batch_normalization_5/' in global_variable.name
                                                                                              or 'batch_normalization_6/' in global_variable.name
                                                                                              or 'batch_normalization_7/' in global_variable.name
                                                                                              or 'batch_normalization_8/' in global_variable.name
                                                                                              or 'batch_normalization_9/' in global_variable.name
                                                                                              or 'batch_normalization_50/' in global_variable.name
                                                                                              or 'batch_normalization_51/' in global_variable.name
                                                                                              or 'batch_normalization_52/' in global_variable.name
                                                                                              or 'batch_normalization_53/' in global_variable.name]
        saver = tf.train.Saver(resnet_classify_vars)
        
        resnet_regress_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='resnet_regress')
        """resnet_regress_vars += [global_variable for global_variable in global_variables_list if 'batch_normalization_36/' in global_variable.name
                                                                                             or 'batch_normalization_37/' in global_variable.name
                                                                                             or 'batch_normalization_38/' in global_variable.name]"""
        #saver_2 = tf.train.Saver(resnet_regress_vars)
        try:
            #sess.run(tf.global_variables_initializer())
            #print("Init weight!")
            saver.restore(sess, "ckpt/resnet_classify.ckpt")
            print("Restore weight!")
            #saver_2.restore(sess, "ckpt/resnet_regress.ckpt")
            #print("Restore weight!")
        except:
            sess.run(tf.global_variables_initializer())
            print("Init all weight!")

        
        def train(epoch):
            for step in range(epoch):
                epoch_start_time = time.time()
                loss_dict = []
                for image, attr, label, image_2 in read_data.get_batch_data(batch_size):
                    _, loss_value = sess.run([train_op, loss], 
                                              feed_dict={data_x: image,
                                                         data_y: label,
                                                         data_x2: image_2,
                                                         user_attr: attr,
                                                         keep_prob: 0.5,
                                                         training: True})
                    loss_dict.append(loss_value)
                ave_loss_value = np.mean(loss_dict)

                print("--step:", step, "--average loss:", ave_loss_value, "--rest time:", (time.time() - epoch_start_time) * (epoch - step - 1))
                if str(ave_loss_value) == "nan":
                    exit(1)
            
            os.system("rm -r ckpt")
            saver.save(sess, "ckpt/resnet_classify.ckpt")
            #aver_2.save(sess, "ckpt/resnet_regress.ckpt")
            print("Save weight!")
    

        def value_calssify(model):
            total_count, error_count = 0, 0
            for image, attr, label, image_2 in read_data.get_batch_data(1, model):
                pre_logit = sess.run(data_logit, 
                                            feed_dict={data_x: image,
                                                        data_y: label,
                                                        data_x2: image_2,
                                                        user_attr: attr,
                                                        keep_prob: 1,
                                                        training: False})
                
                """for logit_index in range(157, len(pre_logit[0])):
                    if pre_logit[0][logit_index] > 0:
                        pre_logit[0][logit_index] *= 2
                    if pre_logit[0][logit_index] < 0:
                        pre_logit[0][logit_index] /= 2"""

                pre_label = np.argmax(pre_logit[0])
                true_label = label[0]

                total_count += 1
                if not pre_label == true_label:
                    error_count += 1 
            print("error_count:", error_count, "total_count:", total_count, "acc_rate:", error_count / total_count)


        def value(model="value"):
            error_count = 0
            total_count = 0
            zsl_count = 0
            train_label_list = list(read_data.train_label2index.keys())
            train_label_list = [train_label + 1 for train_label in train_label_list]
            #print(train_label_list)
            label2embed = list(read_data.label2embed.items())
            #print([label_item for label_item, embed_item in label2embed])
            embeds = [embed_item for label_item, embed_item in label2embed]
            score_bilv_list = []
            score_bilv_list_zsl = []
            for image, attr, label in read_data.get_batch_data(1, model):
                pre_logit = sess.run(data_logit, 
                                    feed_dict={data_x: image * 230,
                                                data_y: embeds,
                                                keep_prob: 1,
                                                training: False})
                
                min_score_index = np.argmin(pre_logit)
                pre_label = label2embed[min_score_index][0]

                true_label = label[0]

                zsl_label, zsl_score = -1, -1
                for index_item, label_embed_item in enumerate(label2embed):
                    label_item, embed_item = label_embed_item
                    if label_item not in train_label_list:
                        if zsl_score == -1 or zsl_score > pre_logit[index_item]:
                            zsl_label, zsl_score = label_item, pre_logit[index_item]

                """if true_label == pre_label or true_label == zsl_label:  # bilv: 1.57
                    #if pre_label == zsl_label:
                        #print("**", end="")
                    print("zsl:", zsl_label, "pre:", pre_label, "true:", true_label, "zsl_score:", zsl_score, "pre_score:", pre_logit[min_score_index], "score_bilv:", pre_logit[min_score_index] / zsl_score)
                    with open(model + '.txt', 'w') as score_file:
                        score_file.write(str(zsl_score) + '\t' + str(pre_logit[min_score_index]) + '\t' + str(pre_logit[min_score_index] / zsl_score) + '\n')"""

                if true_label == pre_label:
                    score_bilv_list.append(pre_logit[min_score_index] / zsl_score)
                if true_label == zsl_label:
                    score_bilv_list_zsl.append(pre_logit[min_score_index] / zsl_score)

                #if -pre_logit[min_score_index] < 0.98 and pre_logit[min_score_index] / zsl_score < 1.57:
                    #pre_label = zsl_label

                total_count += 1
                if true_label == pre_label: 
                    error_count += 1
                if true_label == zsl_label:
                    zsl_count += 1

            print("avg_score_bilv:", np.mean(np.array(score_bilv_list)))
            print("avg_score_bilv:", np.mean(np.array(score_bilv_list_zsl)))
            print(model, "data acc count:", error_count, "zsl acc count:", zsl_count, "data total count:", total_count, "acc_rate:", error_count / total_count)

        
        def test():
            total_count = 0
            train_label_list = list(read_data.train_label2index.keys())
            train_label_list = [train_label + 1 for train_label in train_label_list]
            print(train_label_list)
            print("train_label_list lenth:", len(train_label_list))
            label2embed = list(read_data.label2embed.items())
            print([label_item for label_item, embed_item in label2embed])
            print("train+zsl lenth:", len([label_item for label_item, embed_item in label2embed]))
            embeds = [embed_item for label_item, embed_item in label2embed]
            for image_test, pic_name in read_data.test_image("None"):
                pre_logit = sess.run(data_logit, 
                                     feed_dict={data_x: image_test * 230,
                                                data_y: embeds,
                                                keep_prob: 1,
                                                training: False})
                

                min_score_index = np.argmin(pre_logit)
                pre_label = label2embed[min_score_index][0]

                zsl_label, zsl_score = -1, -1
                for index_item, label_embed_item in enumerate(label2embed):
                    label_item, embed_item = label_embed_item
                    if label_item not in train_label_list:
                        if zsl_score == -1 or zsl_score > pre_logit[index_item]:
                            zsl_label, zsl_score = label_item, pre_logit[index_item]
                
                if -pre_logit[min_score_index] < 0.98 and pre_logit[min_score_index] / zsl_score < 1.57:
                    pre_label = zsl_label

                with open("submit.txt", "a+") as sub_file:
                    sub_file.write(pic_name + '\t' + "ZJL" + str(pre_label) + '\n')

        
        if model=="train":
            while True:
                train(10)
                #value_calssify("train_value")
                value_calssify("value")
                value_calssify("zsl")
                #value("train_value")
                #value("value")
                #value("zsl")
        if model=="value":
            #value_calssify("train_value")
            #value_calssify("value")
            #value("train_value")
            value("value")
            value("zsl")
        if model=="test":
            test()


if __name__ == "__main__":
    batch_size = 256
    model = "train"
    #model = "value"
    #model = "test"

    data_x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    data_x2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 3])
    keep_prob = tf.placeholder(tf.float32)
    training = tf.placeholder(tf.bool)

    user_attr = tf.placeholder(tf.float32, shape=[None, 190, 300])  # remember change for value
    data_y = tf.placeholder(tf.int32, shape=[None])  # remember change for value

    data_x_logit = res_net(data_x, keep_prob, training)
    data_x2_logit = res_net(data_x2, keep_prob, training)

    data_y_logit = mlp_net(data_y, keep_prob, training)

    loss, train_op, cos_similar = loss_func(data_y_logit, data_x_logit, data_x2_logit, user_attr, batch_size)
    data_logit = cos_similar
    #data_logit = data_x_logit

    tf_sess(data_x, data_x2, data_y, user_attr, keep_prob, training, loss, train_op, data_logit, batch_size, model)
