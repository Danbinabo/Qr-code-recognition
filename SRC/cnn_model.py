# 定义CNN Model
import tensorflow as tf
import train as tr


def crack_captcha_cnn(X,w_alpha=0.01, b_alpha=0.1,keep_prob=0.6):
    x = tf.reshape(X, shape=[-1, tr.IMAGE_HEIGHT, tr.IMAGE_WIDTH, 1])

    # w_c1_alpha = np.sqrt(2.0/(IMAGE_HEIGHT*IMAGE_WIDTH)) #
    # w_c2_alpha = np.sqrt(2.0/(3*3*32))
    # w_c3_alpha = np.sqrt(2.0/(3*3*64))
    # w_d1_alpha = np.sqrt(2.0/(8*32*64))
    # out_alpha = np.sqrt(2.0/1024)

    # 3 conv layer1
    #filter定义为3*3*1 输出32个特征 即32个filter
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32])) #特征图:32个
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 0, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob) #keep_prob:dropout保留率

    # 3 conv layer2
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64])) #特征图:64个
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    # 3 conv layer3
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer 1           h = 60 w = 160 / pool后 h ->1/2 w->1/2 这里有三次池化
    # h = 64->32->16->8   w = 160->80->40->20 ===》8 * 20       #1024维向量 = b初始化个数
    w_d = tf.Variable(w_alpha * tf.random_normal([8 * 20 * 64, 1024])) #手动算 前一层链接的参数时多大
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    # Fully connected layer 2 ==》分类 4*10
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, tr.MAX_CAPTCHA * tr.CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([tr.MAX_CAPTCHA * tr.CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out

