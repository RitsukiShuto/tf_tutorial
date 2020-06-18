# Created by RitsukiShuto on 2020/06/11-23:18:11.
# From Temsor Flow tutorial "MNIST"
#
# Coding 'UTF-8'
#
import tensorflow as tf

# MNISTデータセットを入手
mnist = tf.keras.datasets.mnist

# データセットを分割
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0   # ピクセルを正規化

# 学習モデルを構築
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape = (28, 28)),    # 28 * 28ピクセル?
    tf.keras.layers.Dense(128, activation = 'relu'),    # 損失関数
    tf.keras.layers.Dropout(0, 2),
    tf.keras.layers.Dense(10, activation = 'softmax')   # BUG
])

model.compile(optimizer='adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])

model.fit(x_train, y_train, epoch = 5)
model.evaluate(x_test, y_test, verbose = 2)
