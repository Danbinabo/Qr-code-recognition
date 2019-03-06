from captcha.image import ImageCaptcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import train as tr
import tensorflow as tf
import src.cnn_model.crack_captcha_cnn as cnn_model
from src.cnn_model import crack_captcha_cnn

#按图片大小申请占位符
X = tf.placeholder(tf.float32, [None, tr.IMAGE_HEIGHT * tr.IMAGE_WIDTH])  # X结构--固定输入图像的规模
Y = tf.placeholder(tf.float32, [None, tr.MAX_CAPTCHA * tr.CHAR_SET_LEN])  # Y结构--label:4*10个
keep_prob = tf.placeholder(tf.float32)  # dropout

number = [4]
def random_captcha_text(char_set=number, captcha_size=4):
    "生成随机数字"
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def gen_captcha_text_and_image():
    """随机生成数字图片
    :return: 数字，图片
    """
    image = ImageCaptcha()

    captcha_text = random_captcha_text()  # list
    captcha_text = ''.join(captcha_text)  # 转string

    captcha = image.generate(captcha_text)
    # image.write(captcha_text, captcha_text + '.jpg')

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)  # 转 tf可识别的格式
    return captcha_text, captcha_image

def convert2gray(img):
    """转灰度图
    :param img:字典生成的图像
    :return: 灰度图
    """
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img


def text2vec(text,MAX_CAPTCHA,CHAR_SET_LEN):
    "文本转向量"
    text_len = len(text)
    if text_len > MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')

    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    """
    def char2pos(c):  
        if c =='_':  
            k = 62  
            return k  
        k = ord(c)-48  
        if k > 9:  
            k = ord(c) - 55  
            if k > 35:  
                k = ord(c) - 61  
                if k > 61:  
                    raise ValueError('No Map')   
        return k  
    """
    for i, c in enumerate(text):
        idx = i * CHAR_SET_LEN + int(c)
        vector[idx] = 1
    return vector


def vec2text(vec):
    "向量转回文本"
    """
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i #c/63
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx <36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx-  36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    """
    text = []
    char_pos = vec.nonzero()[0]
    for i, c in enumerate(char_pos):
        number = i % 10
        text.append(str(number))

    return "".join(text)


def get_next_batch(batch_size=128):
    "生成训练的一个batch"
    batch_x = np.zeros([batch_size, tr.IMAGE_HEIGHT * tr.IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, tr.MAX_CAPTCHA * tr.CHAR_SET_LEN])

    # 有时生成图像大小不是(60, 160, 3)
    def wrap_gen_captcha_text_and_image():
        while True:
            text, image = gen_captcha_text_and_image()
            if image.shape == (60, 160, 3):
                return text, image

    for i in range(batch_size):
        text, image = wrap_gen_captcha_text_and_image()
        image = convert2gray(image)

        batch_x[i, :] = image.flatten() / 255  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

    return batch_x, batch_y



def crack_captcha(captcha_image):
    """使用训练的模型识别验证码
    :param captcha_image:图片
    :return:识别结果(文字)
    """
    output = crack_captcha_cnn()
    saver = tf.train.Saver()
    keep_prob = 0.6
    with tf.Session() as sess:
        saver.restore(sess, "./model/crack_capcha.model-810")
        predict = tf.argmax(tf.reshape(output, [-1, tr.MAX_CAPTCHA, tr.CHAR_SET_LEN]), 2)
        text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
        text = text_list[0].tolist()
        return text



