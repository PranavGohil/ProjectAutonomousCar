# -*- coding: utf-8 -*-
"""
Created on Sun May 23 14:58:54 2021

@author: Prakhar Mathur
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import numpy 
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
#from tensorflow.keras.layers import Sequential , Conv2D , MaxPooling2D
from tensorflow import keras
import os 
import shutil
from keras.preprocessing.image import load_img,img_to_array
import gym

x_train = []
y_train = [] 
x_validation = []
y_validation = []   

data = np.loadtxt("D:\Python\Model\Regression\Dataset\Training\\10000.txt", delimiter = ",")


print("shape of array", data.shape)

print("First 5 rows:\n", data[:5])


FOLDER_PATH = 'D:\Python\Model\Regression\Dataset\Training'

filenames = os.listdir(FOLDER_PATH)
sorted_filenames = sorted(filenames, key=lambda x: int(str(x.split('.')[0])))
i =0 

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image


for  filename in sorted_filenames:
   #print(filename)
   if filename.endswith(".jpeg"):
    templist = [0 ,0 ,0]
    if (data[i].argmax() == 0 ) :
        templist[0] = -1.0
        y_train.append(templist)
    elif (data[i].argmax() ==1 ):
        templist[0] = 1.0
        y_train.append(templist)
    elif (data[i].argmax() == 2):
        templist[1] = 1.0
        y_train.append(templist)
    else:
        templist[2] = 0.8
        y_train.append(templist)
    x_train.append(load_image(FOLDER_PATH+'\\'+filename))
    i = i+1 
    

y_train = numpy.array(y_train)    
x_train = tf.stack(x_train, axis=0)
#x_train = numpy.array(x_train)    
#print(y_train)
print(x_train.shape)
print(y_train.shape)



# Now we will prepare validation dataset 

FOLDER_PATH = 'D:\Python\Model\Regression\Dataset\Validation'
data = np.loadtxt("D:\Python\Model\Regression\Dataset\Validation\\10000.txt", delimiter = ",")

filenames = os.listdir(FOLDER_PATH)
sorted_filenames = sorted(filenames, key=lambda x: int(str(x.split('.')[0])))
i =0 


for  filename in sorted_filenames:
   #print(filename)
   if filename.endswith(".jpeg"):
    templist = [0 ,0 ,0]
    if (data[i].argmax() == 0 ) :
        templist[0] = -1.0
        y_validation.append(templist)
    elif (data[i].argmax() ==1 ):
        templist[0] = 1.0
        y_validation.append(templist)
    elif (data[i].argmax() == 2):
        templist[1] = 1.0
        y_validation.append(templist)
    else:
        templist[2] = 0.8
        y_validation.append(templist)
    x_validation.append(load_image(FOLDER_PATH+'\\'+filename))
    i = i+1 
  

y_validation= numpy.array(y_validation)    
x_validation = tf.stack(x_validation, axis=0)
#x_train = numpy.array(x_train)    
print(x_validation.shape)
print(y_validation.shape)


def create_q_model():
    inputs = layers.Input(shape=(96, 96, 3,))
    # Convolutions on the frames on the screen
    layer1 = layers.Conv2D(16, 3, strides=1, activation="relu")(inputs)
    layer1 = layers.BatchNormalization()(layer1)
    layer2 = layers.MaxPool2D(pool_size=(2,2),strides=2)(layer1)
    layer3 = layers.Conv2D(32, 3, strides=1, activation="relu")(layer2)
    layer3 = layers.BatchNormalization()(layer3)
    layer4 = layers.MaxPool2D(pool_size=(2, 2), strides=2)(layer3)
    layer5 = layers.Flatten()(layer4)
    layer5 = layers.Dropout(0.5)(layer5)
    layer6 = layers.Dense(64, activation="relu")(layer5)
    act = layers.Dense(3,activation = None)(layer6)
    return keras.Model(inputs=inputs, outputs=act)

model = create_q_model()
model.compile(optimizer="adam",
              loss = "mse",
             metrics=["acc"],)
checkpoint_filepath = 'tmp\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
history = model.fit(x_train, y_train, batch_size= 30, epochs=20,validation_data=(x_validation,y_validation) , callbacks =  [model_checkpoint_callback])


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()



model.load_weights(checkpoint_filepath)

model.save('SavedModel_3_prakhar')

model = keras.models.load_model("SavedModel_3_prakhar")

env=gym.make('CarRacing-v0')

state = env.reset()
for i in range(5000):

    env.render()
    
    #action_agent = model.predict(np.array(state).reshape(1, 96, 96, 3))
    #print(probability_model.predict(np.array(state).reshape(1, 96, 96, 3)))
    action_agent = np.array(model.predict(np.array(state).reshape(1, 96, 96, 3)))
    act = []
    print(action_agent[0][0])
    act.append(action_agent[0][0])
    act.append(action_agent[0][1])
    act.append(action_agent[0][2])
    # if random.random()>0.6:
    #     acc = 1
    # else:
    #     acc = 0
    print(act)
    #state,reward,done,_ = env.step(action_agent[0])
    state, reward, done, _ = env.step(act)
    #action_agent = [0 0 0]   action_agent[0][0]=int 0

    if done:
        env.reset()






             
        





