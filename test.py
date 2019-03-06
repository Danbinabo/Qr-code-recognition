import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.pred import gen_captcha_text_and_image,convert2gray
from src.pred import crack_captcha # 测试

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def test():
    IMAGE_HEIGHT = 60
    IMAGE_WIDTH = 160
    char_set = number
    CHAR_SET_LEN = len(char_set)

    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
    plt.show()

    MAX_CAPTCHA = len(text)
    image = convert2gray(image)
    image = image.flatten() / 255

    X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
    Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
    keep_prob = tf.placeholder(tf.float32)  # dropout

    predict_text = crack_captcha(image)
    print("正确: {}  预测: {}".format(text, predict_text))
    
    
