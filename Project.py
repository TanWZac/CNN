#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from glob import glob
import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt


# F:\Monash\Project\archive\IDC_regular_ps50_idx5

# point to the base path and show the number of patients

# In[ ]:


base = "F:/Monash/Project/archive/IDC_regular_ps50_idx5/"
folder = os.listdir(base)
print(len(folder))


# create directory if not created before

# In[ ]:


img_dir = 'Breast_cancer_dir'

if os.path.isdir(img_dir):
    pass
else:
    os.mkdir(img_dir)


# separate images into pos and neg, saving in different folder
# e.g. 10253_idx5_x1001_y1301_class0.png
#        |          |     |     |     |  
#     patient_id    |     |     |     |
#            x-coordinate |     |     |
#                 y-coordinate  |     |
#          cancer class[No 0/ Yes 1]  |
#                                  file type
# In[ ]:


import shutil
pid = folder
path = "F:/Monash/Project/archive/IDC_regular_ps50_idx5/"
for i in pid:
    N = path + str(i) + '/0'
    P = path + str(i) + '/1'
    
    array0 = os.listdir(N)
    array1 = os.listdir(P)
    
    for j in array0:
        start = os.path.join(N, j)
        end = os.path.join(img_dir, j)
        shutil.copyfile(start, end)
        
    for k in array1:
        start = os.path.join(P, k)
        end = os.path.join(img_dir, k)
        shutil.copyfile(start, end)


# In[ ]:


img_dir = os.listdir('Breast_cancer_dir')
data = pd.DataFrame(img_dir, columns = ['img_id'])

def lab(ind):
    l = ind.split('_')
    d = l[4] 
    return d[5]

def pid(ind):
    id = ind.split('_')
    return id[0]


# In[ ]:


data['pid'] = data['img_id'].apply(pid)
data['lab'] = data['img_id'].apply(lab)


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(data, test_size = 0.2, stratify = data['lab'])


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(data, test_size=0.2, random_state=1, stratify = data['lab'])
# 
# X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1) # 0.25 x 0.8 = 0.2

# In[ ]:


X_train.shape


# In[ ]:


X_test.shape


# create train and valid directory to batch process images

# previous attempt had memory error as pc cannot allocate 68GB in a go
# 
# resolve attempt >>> call from directory
# https://stackoverflow.com/questions/62991747/memory-error-while-training-my-model-unable-to-allocate-31-9-gib-for-an-array-w
# https://www.geeksforgeeks.org/cnn-image-data-pre-processing-with-generators/

# In[ ]:


ori = 'origin'
os.mkdir(ori)

# Train directory
train = os.path.join(ori, 'train')
os.mkdir(train)
# pos and neg train data
pos_train = os.path.join(train, 'pos_train')
neg_train = os.path.join(train, 'neg_train')
os.mkdir(pos_train)
os.mkdir(neg_train)

# Valid directory
test = os.path.join(ori, 'test')
os.mkdir(test)
# pos and neg valid data
pos_test = os.path.join(test, 'pos_test')
neg_test = os.path.join(test, 'neg_test')
os.mkdir(pos_test)
os.mkdir(neg_test)


# In[ ]:


os.listdir('origin/train')


# In[ ]:


data['img_id']


# In[ ]:


data.set_index('img_id', inplace =True)


# In[ ]:


def transfer(d, directory, Data, typ):
    direct = str(typ) 
    for i in d:
        try:
            ind = i
            label = Data.loc[i, 'lab']
            if label == '0':
                end = os.path.join(directory, 'neg_'+direct, ind)
            else:
                end = os.path.join(directory, 'pos_'+direct, ind)
            start = os.path.join('breast_cancer_dir', ind)
            shutil.move(start, end)
        except:
            continue


# In[ ]:


Train = list(X_train['img_id'])
Test = list(X_test['img_id'])

transfer(Train, train, data, 'train')
transfer(Test, test, data, 'test')


# In[ ]:


os.listdir("origin/train/pos_train")


# In[ ]:


os.listdir('origin/test/neg_test')


# In[ ]:


'origin/train/pos_train'


# In[ ]:


from keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale = 1./255)

datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.25,
                            rotation_range=10, width_shift_range=0.2,
                            height_shift_range=0.2, zoom_range=0.2, 
                            horizontal_flip=True)


train_datagen = datagen.flow_from_directory('origin/train', target_size=(50, 50),
                                           batch_size=20,
                                            subset = 'training',
                                           class_mode = 'categorical')

val_datagen = datagen.flow_from_directory('origin/train', target_size=(50, 50),
                                           batch_size=20,
                                           class_mode = 'categorical',
                                               subset = 'validation')

test_datagen = test_datagen.flow_from_directory('origin/test', target_size=(50, 50),
                                           batch_size=1,
                                           class_mode = 'categorical',
                                          shuffle = False) 


# Notes:
# After looking up stackoverflow, there is a way to split the dataset to train and valid
# so the original valid folder will be changed to test data
# 
# https://stackoverflow.com/questions/53037510/can-flow-from-directory-get-train-and-validation-data-from-the-same-directory-in

# In[ ]:


import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, RMSprop, Adam


# ![image.png](attachment:image.png)

# In[ ]:


model = Sequential()
model.add(Convolution2D(64, (3,3), activation = 'relu', input_shape = (50,50,3)))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(Convolution2D(64, (3, 3), activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


# In[ ]:


model.build()
model.summary()


# In[ ]:


model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])


# In[ ]:


history = model.fit_generator(train_datagen, validation_data= val_datagen, steps_per_epoch = 177616//64, validation_steps= 44403//64, epochs=20)


# In[ ]:


model.save('C:/Users/Acer/Desktop/Semester 2/FIT3164/CNN', save_format = 'tf')


# In[ ]:


predict = model.evaluate_generator(test_datagen, verbose = 1)


# model.save('C:/Users/Acer/Desktop/Semester 2/FIT3164/Img_recognition', save_format = 'tf')

# In[ ]:


predict[1]


# In[ ]:


import numpy as np
import cv2 as cv

# img = Image.open('C:/Users/Acer/origin/train/pos_train/8863_idx5_x1001_y801_class1.png')
img = cv.imread('C:/Users/Acer/origin/train/pos_train/10253_idx5_x601_y651_class1.png')
# img = cv.imread('C:/Users/Acer/origin/train/neg_train/8863_idx5_x101_y1201_class0.png')
image = cv.resize(img, (50, 50))
image_tensor = tf.cast(img, tf.float32)
image_tensor = image_tensor / 255.
imgs = image_tensor[None, :]
pred = model.predict(imgs)
p = np.argmax(pred, axis=1)
if p[0] == 0:
  print("No Cancer")
elif p[1] == 1:
  print("cancer")


# In[ ]:





# In[ ]:




