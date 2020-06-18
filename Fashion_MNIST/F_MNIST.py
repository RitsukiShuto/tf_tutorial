# Created by RitsukiShuto on 2020/06/18-15:04:32.
# TensorFlow Tutorial 'F_MNIST.py'
#
# coding 'UTF-8'
#
from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlowをインポート
import tensorflow as tf
from tensorflow import keras

# ヘルパーライブラリのインポート
import numpy as np
import matplotlib.pyplot as plt

print('TensorFlow Ver. ' + tf.__version__)

# データセットをロード
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# クラス名を保存
class_names = ['T-shirts/top', 'Trouser', 'Pullover', 'Dress', 'coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# データの観察
# 訓練データ
print(train_images.shape)   # データのフォーマット
print(len(train_labels))    # 訓練用ラベルの数
print(train_labels)         # 訓練用ラベル

# テストデータ
print(test_images.shape)    # データフォーマット
print(len(test_labels))     # ラベル数
print(test_labels)          # ラベル

# データの前処理
# 画像を例示
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# ピクセルを正規化
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize = (10, 10))

# 画像をクラス名付きで表示
for i in range(25):
    plt.subplot(5, 7, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 学習モデルを構築
model = keras.Sequensial([
    keras.layers.Flatten(input_shape = (28, 28)),   # 入力次元数(28 * 28 = 784)
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

