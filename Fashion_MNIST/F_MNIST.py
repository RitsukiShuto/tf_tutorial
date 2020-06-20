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
#plt.show()

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

#plt.show()

# 学習モデルを構築
model = keras.Sequential([
    keras.layers.Flatten(input_shape = (28, 28)),   # 入力次元数(28 * 28 = 784)
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10, activation = 'softmax')
])

# モデルをコンパイル
model.compile(optimizer = 'adam',
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])

# モデルを訓練
model.fit(train_images, train_labels, epochs = 30)

# 正解率を評価
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest accuracy : ', test_acc)

# 予測する
predictions = model.predict(test_images)

print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])

# 10チャンネルをすべてグラフ化
def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap = plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100 * np.max(predictions_array),
                                        class_names[true_label]),
                                        color = color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color = "#77777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# i番目の画像を確認
i = 0
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

# x個のテスト画像、予想されたラベル、正解ラベルを表示
# 正しい予測は青、間違った予測は赤で表示
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

plt.figure(figsize = (2 * 2 * num_cols, 2 * i * 1))

for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions, test_labels)

plt.show()

# テスト用データセットから画像を一枚取り出す
img = test_images[0]
print(img.shape)

# 画像を１枚だけバッチのメンバーにする
img = (np.expand_dims(img, 0))
print(img.shape)

# 予測を行う
predictions_single = model.predict(img)
print(predictions_single)

plot_value_array(0, predictions_single, test_labels)
_ = plt.xticks(range(10), class_names, rotation = 45)

# バッチの中から予測を取り出す
print(np.argmax(predictions_single[0]))