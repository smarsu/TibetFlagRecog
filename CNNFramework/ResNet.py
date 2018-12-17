from tfNet.tf_net import *


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


def res_net50(data_x, training):
    with tf.variable_scope("ResNet"):
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
        data_logit = identity_bottleneck_conv(data_logit, 2048, (512, 512, 2048), training)

    return data_logit, 2048


def res_net(data_x, training):
    """create a resnet network

    Returns:
        data_x: tf.placeholder().  Input of the whole network. A 4-D shape [batch_size, height, width, channel]
        training: tf.placeholder().  True/False. Used for batch_normalization
        data_logit: Tensor.  The predicted output
    """
    with tf.variable_scope("ResNet"):
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

    return data_logit, 512