import cv2
import tensorflow as tf
import numpy as np

Categories=["zero","one","two","three","four","five","six","seven","eight","nine"]
def prepare(filepath):

    IMAGE_SIZE=28
    IMAGE_ARRAY=cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(IMAGE_ARRAY,(IMAGE_SIZE,IMAGE_SIZE))
    return new_array.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)

model=tf.keras.models.load_model("model.h5")
img='1.jpg'
prediction=model.predict([prepare(img)])
print(prediction)
print(Categories[np.argmax(prediction)])


