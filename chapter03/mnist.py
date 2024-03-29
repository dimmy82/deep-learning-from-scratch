import os
import sys
import numpy as np
# cd chapter03
# python mnist.py
# 上記で実行しないといけない
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def run():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)
    print(x_train.shape)
    print(t_train.shape)
    print(x_test.shape)
    print(t_test.shape)

    img = x_train[0]
    label = t_train[0]
    print(label)
    print(img.shape)
    img = img.reshape(28, 28)
    print(img.shape)

    img_show(img)

run()