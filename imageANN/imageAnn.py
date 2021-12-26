import os
import pickle
import pathlib
import seaborn as sn
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image,ImageOps
from IPython.display import display
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay

warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')
rcParams['figure.figsize'] = (18,8)
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
src_img = Image.open('E:\\FCAI\\projects\\Pycharmprojects\\tt\\dog.4994.jpg')
display(src_img)
gray_img = ImageOps.grayscale(src_img)
gray_resized_img = gray_img.resize(size=(100, 100))
img_final = np.ravel(gray_resized_img) / 255.0
print(img_final)
print(len(img_final))
img_final = pd.DataFrame(img_final).transpose()
print(img_final)
print(len(img_final))
os.chdir('E:\FCAI\projects\Pycharmprojects\dataset')
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')

    for i in random.sample(glob.glob('cat*'), 1280):
        shutil.move(i, 'train/cat')
    for i in random.sample(glob.glob('dog*'), 1280):
        shutil.move(i, 'train/dog')
    for i in random.sample(glob.glob('cat*'), 400):
        shutil.move(i, 'valid/cat')
    for i in random.sample(glob.glob('dog*'), 400):
        shutil.move(i, 'valid/dog')
    for i in random.sample(glob.glob('cat*'), 320):
        shutil.move(i, 'test/cat')
    for i in random.sample(glob.glob('dog*'), 320):
        shutil.move(i, 'test/dog')


os.chdir('../../')

train_path = 'E:\\FCAI\\projects\\Pycharmprojects\\dataset\\train'
valid_path = 'E:\\FCAI\\projects\\Pycharmprojects\\dataset\\valid'
test_path = 'E:\\FCAI\\projects\\Pycharmprojects\\dataset\\test'


def process_Image(img_path:str) -> np.array:
    img = Image.open(img_path)
    img = ImageOps.grayscale(img)
    img = img.resize(size=(100,100))
    img = np.ravel(img) / 255.0
    return img

def processes_folder(folder : pathlib.PosixPath) -> pd.DataFrame :
    processed = []
    for img in folder.iterdir() :
        if img.suffix == '.jpg':
            try :
                processed.append(process_Image(img_path=str(img)))
            except Exception as _:
                continue

    #print(processed)
    processed = pd.DataFrame(processed)
    #print(processed)
    processed['class'] = folder.parts[-1]

    return processed

train_cat = processes_folder(folder=pathlib.Path.cwd().joinpath('E:\\FCAI\\projects\\Pycharmprojects\\dataset\\train\\cat'))

train_dog = processes_folder(folder=pathlib.Path.cwd().joinpath('E:\\FCAI\\projects\\Pycharmprojects\\dataset\\train\\dog'))

train_set = pd.concat([train_cat,train_dog], axis=0)

with open('train_set.pkl' , 'wb') as f :
    pickle.dump(train_set,f)

test_cat = processes_folder(folder=pathlib.Path.cwd().joinpath('E:\\FCAI\\projects\\Pycharmprojects\\dataset\\test\\cat'))

test_dog = processes_folder(folder=pathlib.Path.cwd().joinpath('E:\\FCAI\\projects\\Pycharmprojects\\dataset\\test\\dog'))

test_set = pd.concat([test_cat,test_dog], axis=0)

with open('test_set.pkl' , 'wb') as f :
    pickle.dump(test_set,f)
valid_cat = processes_folder(folder=pathlib.Path.cwd().joinpath('E:\\FCAI\\projects\\Pycharmprojects\\dataset\\valid\\cat'))

valid_dog = processes_folder(folder=pathlib.Path.cwd().joinpath('E:\\FCAI\\projects\\Pycharmprojects\\dataset\\valid\\dog'))

valid_set = pd.concat([valid_cat,valid_dog], axis=0)

with open('valid_set.pkl' , 'wb') as f :
    pickle.dump(valid_set,f)


#print(train_set)
train_set = shuffle(train_set).reset_index(drop=True)
valid_set = shuffle(valid_set).reset_index(drop=True)

x_train = train_set.drop('class',axis = 1)
y_train = train_set['class']

x_test = test_set.drop('class',axis = 1)
y_test = test_set['class']
#print(x_test)
#print(len(x_test))
#print(len(test_set[1]))
#print(test_set)
x_valid = valid_set.drop('class',axis = 1)
y_valid = valid_set['class']

#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y_train = le.fit_transform(y_train)
y_train = tf.keras.utils.to_categorical(y_train.factorize()[0], num_classes=2)
#y_test = le.transform(y_test)
y_test = tf.keras.utils.to_categorical(y_test.factorize()[0], num_classes=2)
#y_valid = le.transform(y_valid)
y_valid = tf.keras.utils.to_categorical(y_valid.factorize()[0], num_classes=2)
print(y_test)
tf.random.set_seed(42)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2048, activation='relu'))
model.add(tf.keras.layers.Dense(1024, activation='relu'))

model.add(tf.keras.layers.Dense(512, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))


model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.compile(
    loss= tf.keras.losses.categorical_crossentropy,
    optimizer= tf.keras.optimizers.Adam(),
    metrics= [tf.keras.metrics.BinaryAccuracy(name= 'accuracy')]
)

history = model.fit(x_train,y_train,epochs=100,batch_size=128,validation_data=(x_valid,y_valid))


y_pred = model.predict(x_test)
print(y_pred)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# y_pred = (y_pred>0.5)
print(y_pred)

print(x_train)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix , accuracy_score
# cm = confusion_matrix(y_test[1],y_pred[1])
# print(test_set)
# print(y_test)
# cm = tf.metrics.confusion_matrix(y_test.classes, y_pred)
# print(cm)
# print('Classification Report')
# print(tf.metrics.classification_report(y_test.classes, y_pred))

#print(cm)

a=history.history['accuracy']
b=len(a)
z=a[b-1]
print("accuracy:",z)
a=history.history['val_loss']
b=len(a)
z=a[b-1]
print("validation loss",z)
a=history.history['loss']
b=len(a)
z=a[b-1]
print("loss",z)
a=history.history['val_accuracy']
b=len(a)
z=a[b-1]
print("validation accuracy",z)




plt.plot(np.arange(1,101),history.history['loss'], label= 'Training loss')
plt.plot(np.arange(1,101),history.history['val_loss'], label= 'Validation loss')
plt.plot(np.arange(1,101),history.history['accuracy'], label= 'Accuracy')
plt.plot(np.arange(1,101),history.history['val_accuracy'], label= 'validiation Accuracy')
plt.title('Training and Validation loss and Accuracy', size=20)
plt.legend();
plt.show();


























