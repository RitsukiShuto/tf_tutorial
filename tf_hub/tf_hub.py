# Created by RitsukiShuto on 2020/06/20-16:13:48.
# TensorFlowチュートリアル 'th_fub.py'
#
# Coding 'UTF-8'
#
# ライブラリをインポート
import numpy as np
import tensorflow as tf

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# バージョン情報を表示
#print("---DBG INFO---")
#print("Ver. :", tf.__version__)
#print("Eager mode: ", tf.executing_eagerly())
#print("Hub ver. :", hub.__version__)
#print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
#print("---END---")

# データセットを分割
# 訓練データを60%と40%に分割, 15,000件
# 訓練 => 10,000 テスト => 25,000
train_data, validation_data, test_data = tfds.load(
    name = "imdb_reviews",
    split = ('train[:60%]', 'train[60%:]', 'test'),
    as_supervised = True)

# データを探索
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch)     # 訓練データのデータを表示
print(train_labels_batch)       # 訓練データのデータラベルを表示

# 学習モデルを構築
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape = [],
                            dtype = tf.string, trainable = True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dense(1))

model.summary()

model.compile(optimizer = 'adam',
                loss = tf.keras.losses.BinaryCrossentropy(from_logits = True),
                metrics = ['accuracy'])

# 訓練
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs = 20,
                    validation_data = validation_data.batch(512),
                    verbose = 1)

# 学習モデルを評価
results = model.evalate(test_data.batch(512), verbose = 2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
