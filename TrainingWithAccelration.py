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

'''x_train = []
y_train = [] 
x_validation = []
y_validation = []  


def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image)
    return image

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




data = np.loadtxt("D:\Python\Model\Regression\Dataset\Training\\10000.txt", delimiter = ",")
FOLDER_PATH = 'D:\Python\Model\Regression\Accelration_Dataset'
filenames = os.listdir(FOLDER_PATH)
sorted_filenames = sorted(filenames, key=lambda x: int(str(x.split('.')[0])))
i =0 

for  filename in sorted_filenames:
   #print(filename)
   if filename.endswith(".jpeg"):
    templist = [0 ,1 ,0]
    y_train.append(templist)
    x_train.append(load_image(FOLDER_PATH+'\\'+filename))
    i = i+1 
    if(i>100) :
        break

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
print(y_train.shape) '''

'''y_train= numpy.array(y_train)    
x_train = tf.stack(x_train, axis=0)
#x_train = numpy.array(x_train)    
print(y_train.shape)
print(x_train.shape)'''




'''model = keras.models.load_model("SavedModel_Pranav_3")

history = model.fit(x_train, y_train, batch_size= 30, epochs=3,validation_data=(x_validation,y_validation) )

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save('SavedModel_MAE_Pranav_Acc')'''

model = keras.models.load_model("SavedModel_MAE_Pranav_Acc")

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








