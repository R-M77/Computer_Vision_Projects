"""Multiple neurons for classifying more than 1 object/thing"""

# Classify handwritten digits 0 to 9

from matplotlib import cm
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""import dataset"""
data = tf.keras.datasets.mnist
(train_Images, train_Labels), (test_Images, test_Labels) = data.load_data()

"""normalize images from 0 to 1"""
train_Images = train_Images/255.0
test_Images = test_Images/255.0

"""flatten layers from 28x28 pixels to X by 1"""
flat_Layer = tf.keras.layers.Flatten(input_shape=(28,28))
"""define layers 1 and 2"""
layer_1 = tf.keras.layers.Dense(20, activation = tf.nn.relu)
layer_2 = tf.keras.layers.Dense(10, activation = tf.nn.softmax)

# ReLU changes any output that is less than 0 to 0. Commonly used in Dense
    # so neurons dont cancel each other out
# softmax finds the biggest value in the output layer
    # highest output value sets the probability of that classification

"""build sequential dense model"""
model = tf.keras.models.Sequential([flat_Layer,layer_1,layer_2])

"""compile model for classification. adam optimizer for varibale/adaptive LR"""
"""categorical loss function for classification of categories"""
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_Images,train_Labels,
validation_data = (test_Images,test_Labels), epochs = 20)

"""call model predictions"""
classifications = model.predict(test_Images)
print(classifications[0])
print(test_Labels[0])

"""define class names, plot images"""
class_Names = ['0','1','2','3','4','5','6','7','8','9']
def plotImage (i, predict_Array, true_Label, img):
    true_Label, img = true_Label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predictions_Label = np.argmax(predict_Array)
    if predictions_Label == true_Label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel('{} {:2.0f}% ({})'.format(class_Names[predictions_Label],
    100*np.max(predict_Array),class_Names[true_Label],color=color))

"""plot prediction values and compare to true labels"""
def plot_Value(i, predictions_Array, true_Label):
    true_Label = true_Label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10),predictions_Array, color='#777777')
    plt.ylim([0,1])
    predicted_Label = np.argmax(predictions_Array)

    thisplot[predicted_Label].set_color('red')
    thisplot[true_Label].set_color('green')

"""output plots with images and predicted values | validation """
num_rows = 10
num_cols = 10
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plotImage(i, classifications[i], test_Labels, test_Images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_Value(i, classifications[i], test_Labels)
plt.tight_layout()
plt.show()