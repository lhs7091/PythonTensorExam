import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[4], [3]],
                  [[1], [2]]]], dtype=np.float32)

pool = tf.nn.max_pool(image, ksize=[1,2,2,1], strides=[1,1,1,1], padding='SAME')
print(pool.shape)
print(pool.eval())