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
print("---DBG INFO---")
print("Ver. :", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub ver. :", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")
print("---END---")
