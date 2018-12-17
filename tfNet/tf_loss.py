import tensorflow as tf

def smoothL1(logits, labels):
    """A stronger loss func for sum_square"""
    diff = logits - labels

    def less_loss(inp):
        oup = 0.5 * inp * inp
        return oup
    
    def great_loss(inp):
        oup = tf.abs(inp) - 0.5
        return oup

    smooth_loss = tf.where(tf.greater(tf.abs(diff), 1.0), great_loss(diff), less_loss(diff))
    return smooth_loss


def L2_weight_decay(weight_decay):
    """Regularization"""
    global_variables_list = tf.global_variables()
    _weight_vars = [global_variable for global_variable in global_variables_list if '_weight' or '_bias' in global_variable.name]

    weight_loss = 0
    for _weight in _weight_vars:
        weight_loss += weight_decay * tf.nn.l2_loss(_weight)
    return weight_loss


def smooth_loss(loss, tag, smooth_factor):
    loss = tf.where(tf.equal(tag, 1), loss, loss * smooth_factor)
    return loss


def hard_sample_mining(loss, k):
    loss = tf.reshape(loss, [-1])
    loss = tf.nn.top_k(loss, k)[0]
    return loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
        List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def get_tower_grad(opt, loss):
    grad_arr = opt.compute_gradients(loss)
    return grad_arr


def apply_grad(opt, grad_concat):
    train_op = opt.apply_gradients(grad_concat)
    return train_op
