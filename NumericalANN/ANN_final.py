#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[15]:


import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
data = pd.read_csv('C:/Users/GAMING/OneDrive/Desktop/sel/sel/heart.csv')
#print(data)
x = data.iloc[:,0:-1].values
y = data.iloc[: , -1].values
#print(x)
#print(y)
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
le = LabelEncoder()
x[:,1] = le.fit_transform(x[:,1])
le = LabelEncoder()
x[:,8] = le.fit_transform(x[:,8])
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])
le = LabelEncoder()
x[:,6] = le.fit_transform(x[:,6])
le = LabelEncoder()
x[:,10] = le.fit_transform(x[:,10])


#train = ['Age','RestingBP','Cholesterol','MaxHR']

print(x)

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=1)

sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(X_train,Y_train,batch_size=32,epochs=100)
Y_pred = ann.predict(X_test)
Y_pred = (Y_pred>0.5)
print(np.concatenate((Y_pred.reshape(len(Y_pred),1), Y_test.reshape(len(Y_test),1)),1))
from sklearn.metrics import confusion_matrix , accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
print(X_train)

print(100*accuracy_score(Y_test,Y_pred))
predictions = ann.predict_proba(X_test)


#Roc Curve
fpr1, tpr1, _ = roc_curve(Y_test, predictions[:,0])
plt.plot(fpr1, tpr1)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC curve')
plt.show()


#Loss Curve
history = ann.fit(X_train, Y_train, validation_split=0.25, batch_size = 32, epochs = 100)
loss_train = history.history['loss']
loss_val = history.history['val_loss']
#epochs = range(1,35)
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




