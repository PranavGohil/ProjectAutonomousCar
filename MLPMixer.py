# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 11:00:36 2021

@author: Prakhar Mathur
"""

# Copyright 2021 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# our hidden size intially 4
#no of channels 3 

from typing import Any
import einops.layers.torch as tensorflow_einops
#import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds

import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models
#from tensorflow.keras.layers import Sequential , Conv2D , MaxPooling2D
from tensorflow import keras
import os 
import shutil
from keras.preprocessing.image import load_img,img_to_array
import gym
from PIL import Image
from numpy import asarray
import cv2 
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import datasets, layers, models
#from einops.layers.keras import Rearrange


x_train = []
y_train = [] 
x_validation = []
y_validation = []   

def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image


data = np.loadtxt("D:\Python\Model\Regression\Dataset\Training\\10000.txt", delimiter = ",")


print("shape of array", data.shape)

print("First 5 rows:\n", data[:5])


FOLDER_PATH = 'D:\Python\Model\Regression\Dataset\Training'

filenames = os.listdir(FOLDER_PATH)
sorted_filenames = sorted(filenames, key=lambda x: int(str(x.split('.')[0])))
i =0 


d =0
for  filename in sorted_filenames:
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
    Current_arr = load_image(FOLDER_PATH+'\\'+filename)
    #temp_arr = np.concatenate((Current_arr , fourth_array , third_array, second_array , first_array) , axis = 2)
    x_train.append(Current_arr)
    #second_array = third_array
    #third_array = fourth_array
    #fourth_array = Current_arr
    i = i+1 
    #if(i>3):
      #  break
 
 
first_array = np.zeros((96,96,1))
second_array = np.zeros((96,96,1))
third_array = np.zeros((96,96,1))
fourth_array = np.zeros((96,96,1))
i = 0

'''data = np.loadtxt("D:\Python\Model\Regression\Dataset\ACC\\10000.txt", delimiter = ",")   
 
FOLDER_PATH = 'D:\Python\Model\Regression\Dataset\ACC'

filenames = os.listdir(FOLDER_PATH)
sorted_filenames = sorted(filenames, key=lambda x: int(str(x.split('.')[0])))
i =0 
for  filename in sorted_filenames:
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
    Current_arr = load_image(FOLDER_PATH+'\\'+filename)
    #temp_arr = np.concatenate((Current_arr , fourth_array , third_array, second_array , first_array) , axis = 2)
    x_train.append(Current_arr)
    #second_array = third_array
    #third_array = fourth_array
    #fourth_array = Current_arr
    i = i+1 
    #if(i>100):
        #break
    
    
 '''   
y_train = np.array(y_train)
#x_data = x_train    
x_train = tf.stack(x_train, axis=0)
#x_train = numpy.array(x_train)    
#print(y_train)
#print(x_train.shape)
#print(y_train.shape)



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
    Current_arr = load_image(FOLDER_PATH+'\\'+filename)
    #temp_arr = np.concatenate((Current_arr, fourth_array , third_array, second_array , first_array) , axis = 2)
    x_validation.append(Current_arr)
    #second_array = third_array
    #third_array = fourth_array
    #fourth_array = Current_arr
    i = i+1 
  

y_validation= np.array(y_validation)    
x_validation = tf.stack(x_validation, axis=0)
#x_train = numpy.array(x_train)    
#print(x_validation.shape)
#print(y_validation.shape)








def MLPMIXER():
            inputs = layers.Input(shape=(96, 96, 3,))
            x = layers.Conv2D( 3, 6 , strides=6)(inputs)
            x = layers.Reshape((256,3))(x)
            #x = tensorflow_einops.Rearrange('n h w c -> n (h w) c')(x)
            #x = MixerBlock_Keras( 256 , 3)(x)
            y = layers.Permute((2,1))(x)
            y = layers.Dense(256)(y)
            y = layers.Activation('gelu')(y)
            y = layers.Dense(256)(y)
            y =  layers.Permute((2,1))(y)
            x = x + y
            y = layers.BatchNormalization()(x)
            y = layers.Dense(3)(y)
            y = layers.Activation('gelu')(y)
            y = layers.Dense(3)(y)
            x = x + y
            x = layers.BatchNormalization()(x)
            y = layers.Permute((2,1))(x) #second block
            y = layers.Dense(256)(y)
            y = layers.Activation('gelu')(y)
            y = layers.Dense(256)(y)
            y =  layers.Permute((2,1))(y)
            x = x + y
            y = layers.BatchNormalization()(x)
            y = layers.Dense(3)(y)
            y = layers.Activation('gelu')(y)
            y = layers.Dense(3)(y)
            y = x + y 
            y = tf.reduce_mean(y , axis = 1)
            act = layers.Dense(3)(y)
            return keras.Model(inputs , act)
        

model = MLPMIXER()
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
history = model.fit(x_train, y_train, batch_size= 30, epochs=10,validation_data=(x_validation,y_validation) , callbacks =  [model_checkpoint_callback])

model.load_weights(checkpoint_filepath)

model.save('SavedModel_MLPMixer_3')

'''model = keras.models.load_model("SavedModel_MLPMixer_2")

checkpoint_filepath = 'tmp\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_acc',
    mode='max',
    save_best_only=True)
history = model.fit(x_train, y_train, batch_size= 30, epochs=10,validation_data=(x_validation,y_validation) , callbacks =  [model_checkpoint_callback])

model.load_weights(checkpoint_filepath)
model.save('SavedModel_MLPMixer_acc_2')'''



env=gym.make('CarRacing-v0')

state = env.reset()
for i in range(5000):

    env.render()
    
    #action_agent = model.predict(np.array(state).reshape(1, 96, 96, 3))
    #print(probability_model.predict(np.array(state).reshape(1, 96, 96, 3)))
    action_agent = np.array(model.predict(np.array(state).reshape(1, 96, 96, 3)))
    act = []
    act.append(action_agent[0][0])
    act.append(action_agent[0][1])
    act.append(action_agent[0][2])
    # if random.random()>0.6:
    #     acc = 1
    # else:
    #     acc = 0
    '''if(act[2] > 0.2):
        act[2] = act[2]/2
        
    if (act[1] > 0 and act[1] < 0.1) :
        act[1] = act[1]*20
    print(act)'''
    #state,reward,done,_ = env.step(action_agent[0])
    state, reward, done, _ = env.step(act)
    #action_agent = [0 0 0]   action_agent[0][0]=int 0

    if done:
        env.reset()




