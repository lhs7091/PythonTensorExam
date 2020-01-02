# build graph using TF operations

#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as tf

hello = tf.constant('hello!! tensorflow')

sess = tf.Session()


print(sess.run(hello))




