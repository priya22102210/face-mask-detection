#!/usr/bin/env python
# coding: utf-8

# In[57]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
#Generate batches of tensor image data with real-time data augmentation.


# In[58]:


from tensorflow.keras.applications import MobileNetV2


# In[59]:


from tensorflow.keras.layers import AveragePooling2D


# In[60]:


from tensorflow.keras.layers import Dropout


# In[61]:


from tensorflow.keras.layers import Flatten


# In[62]:


from tensorflow.keras.layers import Dense


# In[63]:


from tensorflow.keras.layers import Input


# In[64]:


from tensorflow.keras.models import Model


# In[65]:


from tensorflow.keras.optimizers import Adam
#Optimizer that implements the Adam algorithm.

#Adam optimization is a stochastic gradient descent 
#method that is based on adaptive estimation of first-order and second-order moments.


# In[66]:


from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
#meant to adequate your image to the format the model requires.


# In[67]:


from tensorflow.keras.preprocessing.image import img_to_array


# In[68]:


from tensorflow.keras.preprocessing.image import load_img
#Loads an image into PIL format


# In[69]:


from tensorflow.keras.utils import to_categorical
#Converts a class vector (integers) to binary class matrix.


# In[70]:


from sklearn.preprocessing import LabelBinarizer
#Function to perform the transform operation of LabelBinarizer with fixed classes.


# In[71]:


from sklearn.model_selection import train_test_split


# In[72]:


from sklearn.metrics import classification_report
#build a text report showing the main classification metrics.


# In[73]:


from imutils import paths


# In[74]:


import matplotlib.pyplot as plt


# In[75]:


import numpy as np


# In[76]:


import os


# In[77]:


INIT_LR=1e-4
EPOCHS =20
BS=32


# In[78]:


DIRECTORY=r"C:/Users/EB562TS/Downloads/deeplearning/observationsmaster/experiements/data"


# In[79]:


CATEGORIES=["with_mask","without_mask"]


# In[80]:


print(" {INFO} loading images....")


# In[81]:


data=[]
labels=[]


# In[82]:


for category in CATEGORIES:
    path=os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path=os.path.join(path, img)
        image=load_img(img_path,target_size=(224,224))#resizing all the images
        image=img_to_array(image)#converting to arrays
        image=preprocess_input(image)
        
        data.append(image)
        labels.append(category)
        
        
        


# In[83]:


lb=LabelBinarizer()#labels being converted into binary format
labels=lb.fit_transform(labels)
labels=to_categorical(labels)


# In[84]:


data=np.array(data,dtype="float32")


# In[85]:


labels=np.array(labels)


# In[86]:


(trainX,testX,trainY,testY)=train_test_split(data,labels,test_size=0.20,stratify=labels,random_state=42)


# In[87]:


#constructing the training image generator for data augmentation
aug=ImageDataGenerator(rotation_range=20,
                       zoom_range=0.15,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       shear_range=0.15,
                       horizontal_flip=True,fill_mode="nearest")


# In[88]:


#loading the mobilenetv2 network,ensuring the head FC 
#layer sets are left off


# In[89]:


#loading themobilenet with pretrained imagesrate weights leading the head of networks
baseModel=MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))


# In[90]:


#construct the head of the model that will
#be placed on top of the base model


# In[92]:


headModel=baseModel.output
headModel=AveragePooling2D(pool_size=(7,7))(headModel)
headModel=Flatten(name="flatten")(headModel)
headModel=Dense(228,activation="relu")(headModel)
headModel=Dropout(0.5)(headModel)
headModel=Dense(2,activation="softmax")(headModel)
#we are constructing a new fc head  and 
#appending into the base in case of old head


# In[93]:


#place the head FC model on top of the base model,this
#will become the actuall model we will train


# In[96]:


model=Model(inputs=baseModel.input,outputs=headModel)
#freezing th ebase layers of the network, the weights of 
#these base layers wil not be updated during the process
#of bp whereas the headlayer weights will be tuned


# In[97]:


#we will finetune our mobilenetv2 


# In[98]:


for layer in baseModel.layers:
    layer.trainable=False


# In[99]:


#after defining all the hidden or output layers we have to compile them


# In[100]:


print("{INFO} calling model....")


# In[104]:


#compiling
opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=opt,metrics=["accuracy"])


# In[108]:


#train the head of the network
print("{INFO} training head....")
H=model.fit(aug.flow(trainX,trainY,batch_size=BS),
            steps_per_epoch=len(trainX)//BS,
            validation_data=(testX,testY),
            validation_steps=len(testX)//BS,
            epochs=EPOCHS)


# In[109]:


#make predictions on the testing set
print("{INFO} evaluating network...")
predIdxs=model.predict(testX,batch_size=BS)


# In[110]:


#for each image in the testing set we need to find
#index of the label with corresponding largest predicted probability


# In[111]:


predIdxs=np.argmax(predIdxs,axis=1)


# In[112]:


#show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1),predIdxs,target_names=lb.classes_))


# In[114]:


print("{INFO} sving mask detector model....")
model.save("mask_detector.model",save_format="h5")


# In[116]:


#import pickle
#save_model=model.dump("something.pk")


# In[120]:


#plot the training loss and  accuracy
N=EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"],label="train_loss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"],label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")


# In[ ]:


#accuracy is good

