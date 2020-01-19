import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()
image = np.array([[[[1], [2], [3]],
                  [[4], [5], [6]],
                  [[7], [8], [9]]]], dtype=np.float32)

#print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys') #(1, 3, 3, 1) (the number of images, x, y, the number of colors)
#plt.show()

print("Image shape : ", image.shape)
W = tf.constant([[[[1.]], [[1.]]],
                 [[[1.]], [[1.]]]]) # (2, 2, 1, 1) (x, y, the number of colors, the number of filters)
print("Weight shape : ", W.shape)
conv2d = tf.nn.conv2d(image, W, strides=[1,1,1,1], padding='SAME') # stirdes=[1, x, y, 1]
conv2d_image = conv2d.eval()
print("conv2d shape : ", conv2d.shape)

# to display the result of image
conv2d_image = np.swapaxes(conv2d_image, 0, 3)
for i, one_image in enumerate(conv2d_image):
    print(one_image.reshape(3,3))
    plt.subplot(1,2,i+1), plt.imshow(one_image.reshape(3,3), cmap='gray')

plt.show()