#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf


# #Loading  "MNIST data set"
# ##Containing Training samples = 60,000  Testing samples = 10,000
# ###Tensflow already contain MNSIT data set which can be loaded using keras

# In[2]:


mnist = tf.keras.datasets.mnist## this is basically handwritten characters based on 28x28 sized images of 0 to 9


# After loading the MNSIT data, Divide into train and test datasets

# In[3]:


## unpacking the dataset into train and test dataset
(x_train,y_train),(x_test,y_test)=mnist.load_data()


# In[4]:


x_train.shape


# In[5]:


## just check the graph , how data looks like
import matplotlib.pyplot as plt
plt.imshow(x_train[0])
plt.show() ## in order to excute the graph
## however we don't know whether its color image or binary image
## so inorder to plot it change the configuration
plt.imshow(x_train[0],cmap = plt.cm.binary)


# # Checking the values of each pixel
# ### Before Normalization

# In[7]:


print(x_train[0])


# ##As images are in Grey level(1 channel==> 0 to 255), not Colored(RGB)
# ###Normalizing the data | Pre-Processing Step

# In[6]:


## you might have noticed that, its gray image and all values varies from 0 to 255
### in order to normalixe it 
x_train = tf.keras.utils.normalize(x_train,axis = 1)
x_test  = tf.keras.utils.normalize(x_test,axis = 1)
plt.imshow(x_train[0], cmap = plt.cm.binary)


# ## After Normalization

# In[7]:


print(x_train[0]) ## you can see all values are now normalized


# In[10]:


print(y_train[0]) ## just to check we have labels inside our network


# ## Resizing imag to make it suitable for apply Convolution operation

# In[8]:


import numpy as np
IMG_SIZE=28
x_trainr = np.array(x_train).reshape(-1,IMG_SIZE,IMG_SIZE,1) ### Increasing one dimension for kernel operation
x_testr= np.array(x_test).reshape(-1,IMG_SIZE,IMG_SIZE,1) ### Increasing one dimension for kernel operation
print("Training Samples dimension",x_trainr.shape)
print("Testing Samples dimension",x_testr.shape)


# # Creating a Deep neural Network
# ### Training on 60,000 samples of MNIST handwritten dataset

# In[9]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout, Activation,Flatten,Conv2D,MaxPooling2D


# In[10]:


### Creating a neural network now
model = Sequential()

### First Convolution Layers 0 1 2 3 (60000,28,28,1) 28-3+1=26x26
model.add(Conv2D(64,(3,3),input_shape = x_trainr.shape[1:])) ## only for first convolution layer to metion input size
model.add(Activation("relu")) ## activation function to make it non-linear,<0,remove, >0
model.add(MaxPooling2D(pool_size=(2,2)))## Maxpooling single maxmum value of 2x2,

### 2nd Convolution Layer 26-3+1 = 24x24
model.add(Conv2D(64,(3,3))) ## 2nd Convolution layer
model.add(Activation("relu")) ## activation function 
model.add(MaxPooling2D(pool_size=(2,2)))## Maxpooling single

## 3rd convolution layer  
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#### Fully connected Layer # 1 20x20 = 400
model.add(Flatten()) ## before using fully comnnected layer,need to be flatten so that
model.add(Dense(64)) # neural network
model.add(Activation("relu"))

### fully Connected layer #2
model.add(Dense(32))
model.add(Activation("relu"))

### Last Fully connected layer, output must be equal to number of classes, 10 (0-9)
model.add(Dense(10)) ##this last dense layer must be equal to 10
model.add(Activation('softmax')) ### activation function is changed to softmax(class probabilties)


# In[14]:


model.summary()


# In[11]:


print("Total Training Samples= ",len(x_train))


# In[12]:


model.compile(loss = "sparse_categorical_crossentropy",optimizer="adam",metrics=['accuracy'])


# In[28]:


model.fit(x_trainr,y_train,epochs=10,validation_split=0.3) ## Training my model


# In[29]:


###evaluating on testing data set MNIT
test_loss,test_acc = model.evaluate(x_testr,y_test)
print("Test loss on 10,000 test samples",test_loss)
print("validaation Accuracy on 10,000 test samples",test_acc)


# In[4]:


model = tf.keras.models.load_model("mnist-digit-model.h5")


# In[31]:



prediction=model.predict([x_testr])


# In[20]:


print(prediction)


# In[32]:


print(np.argmax(prediction[0]))


# In[47]:


plt.imshow(x_test[3])


# In[34]:


print(np.argmax(prediction[128]))


# In[35]:


plt.imshow(x_test[128])


# In[10]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[37]:


#img=cv2.imread('digit2.png')


# In[38]:


#plt.imshow(img)


# In[39]:


#img.shape


# In[40]:


#gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# In[41]:


#gray.shape


# In[42]:


#resized=cv2.resize(gray,(28,28),interpolation=cv2.INTER_AREA)


# In[43]:


#resized.shape


# In[15]:


img=cv2.imread(f"digit2.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#ret,img = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
img = cv2.resize(img, (28,28))
img=np.invert(np.array([img]))
prediction =model.predict(img)
print(np.argmax(prediction))
plt.imshow(img[0], cmap=plt.cm.binary)
plt.show()


# In[21]:


model.save("mnist-digit-model.h5")

