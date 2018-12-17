import tensorflow as tf


def conv_net(inp, filter, strides, padding="SAME", add_b=False):
    """create a conv layer"""
    conv_w = tf.Variable(tf.truncated_normal(shape=filter, stddev=0.01), name='_weight')
    conv_h = tf.nn.conv2d(inp, conv_w, strides, padding)
    if add_b:
        conv_b = tf.Variable(tf.constant(0.0, shape=[filter[-1]]), name='_bias')
        conv_h += conv_b
    return conv_h


def dense_net(inp, shape, add_b=False, ret_wb=False):
    """create a dense layer"""
    dense_w = tf.Variable(tf.truncated_normal(shape=shape, stddev=0.0001), name='_weight')
    dense_h = tf.matmul(inp, dense_w)
    if add_b:
        dense_b = tf.Variable(tf.constant(0.0, shape=[shape[-1]]), name='_bias')
        dense_h += dense_b
    if ret_wb:
        return dense_h, dense_w, dense_b if add_b else None 
    return dense_h


def batch_normalization(inp, training=False):
    """batch_normalize for next layer's input"""
    inp_norm = tf.layers.batch_normalization(inp, training=training)
    return inp_norm


def actv_net(inp, model="leaky_relu"):
    """actvation function"""
    if model == "leaky_relu":
        inp_actv = tf.maximum(0.1 * inp, inp)
    return inp_actv


def max_pool(inp, w_shape, strides, padding="SAME"):
    """max pool layer"""
    pool_h = tf.nn.max_pool(inp, w_shape, strides, padding)
    return pool_h


def avg_pool(inp, w_shape, strides, padding="SAME"):
    """avg pool layer"""
    pool_h = tf.nn.avg_pool(inp, w_shape, strides, padding)
    return pool_h


def global_avg_pool(inp, channel):
    """global avg pool"""
    pool_h = tf.reduce_mean(inp, axis=[1,2])
    pool_h = tf.reshape(pool_h, [-1, 1, 1, channel])
    return pool_h


def global_max_pool(inp, channel):
    """global max pool"""
    pool_h = tf.reduce_max(inp, axis=[1,2])
    pool_h = tf.reshape(pool_h, [-1, 1, 1, channel])
    return pool_h


def drop_out(inp, keep_prob):
    """drop out layer"""
    drop_h = tf.nn.dropout(inp, keep_prob)
    return drop_h


def dense_pool(inp, w_shape, strides, mode, padding="SAME"):
    """dense pool"""
    if mode == "max":
        pool_func = max_pool
    if mode == "avg":
        pool_func = avg_pool
    
    strip_h, strip_w = strides[1:3]
    fmap_h, fmap_w = inp.get_shape()[1:3]
    dense_pool_out = 0
    for i in range(strip_h):
        tail_i = i + fmap_h 
        for j in range(strip_w):
            tail_j = j + fmap_w
            crop_inp = inp[:, i:tail_i, j:tail_j, :]
            pool_out = pool_func(crop_inp, w_shape, strides, padding)
            dense_pool_out = dense_pool_out + pool_out

    dense_pool_out = dense_pool_out / (strip_h * strip_w)
    return dense_pool_out
