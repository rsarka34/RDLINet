import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from pathlib import Path
import os
import wave
import keras
import seaborn as sns
import librosa
import soundfile as sf
from google.colab import drive
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from numpy import mean
from numpy import std
import tensorflow as tf
from numpy import dstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tqdm import tqdm
import os
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
print(keras.__version__)
!pip install mat73
import mat73

drive.mount('/content/gdrive')

import scipy.io as sio
sig_dict = sio.loadmat('/content/gdrive/MyDrive/7clss_orig_dim/imgs_64_38.mat')
X=sig_dict['train_img']

labels=pd.read_excel('/content/gdrive/MyDrive/7clss_orig_dim/7class_labels.xlsx',header=None)
labels.columns=["Lung Sound"]
class_label_onehot=pd.get_dummies(labels)
Y=np.array(class_label_onehot);
print(X.shape)
print(Y.shape)

def RDLINet(dim, output_neurons, output_activation):
    print("\nTRAINING ON RDLINet:-")

    def block(x, filters, reps):
        for _ in range(reps):
            # for low-level features
            t1 = Conv2D(filters[0], kernel_size = (1,1))(x)
            t1 = LeakyReLU()(t1)

            t2 = DepthwiseConv2D(kernel_size = (3,3), strides = 1, padding = 'same')(x)
            t2 = LeakyReLU()(t2)
            t2 = Conv2D(filters[1], kernel_size = (1,1))(t2)
            t2 = LeakyReLU()(t2)

            t3 = DepthwiseConv2D(kernel_size = (5,5), strides = 1, padding = 'same')(x)
            t3 = LeakyReLU()(t3)
            t3 = Conv2D(filters[2], kernel_size = (1,1))(t3)
            t3 = LeakyReLU()(t3)

            t4 = MaxPool2D(pool_size = (3,3), strides = 1, padding = 'same')(x)
            t4 = Conv2D(filters[3], kernel_size = (1,1))(t4)
            t4 = LeakyReLU()(t4)

            x = Concatenate()([t1, t2, t3, t4])

        return x


    input = Input(shape = dim)

    k = 16

    x = Conv2D(filters = k, kernel_size = (3,3), strides = 2, padding = 'same')(input)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (3,3), strides = 2, padding = 'same')(x)

    x = DepthwiseConv2D(kernel_size = (3,3), strides = 1, padding = 'same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(filters = 2*k, kernel_size = (1,1))(x)
    x = LeakyReLU()(x)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)

    x = block(x, [k, k, k, k], reps = 2)
    x = MaxPool2D(pool_size = (2,2), strides = 2)(x)

    x = GlobalAveragePooling2D()(x)
    t1= Dense(15, 'sigmoid')(x)
    t2= Dense(15,'linear')(x)
    t3=Multiply()([t1,t2])

    #x = Dropout(0.3)(x)
    #x=  Dense(40, 'relu')(x)
    #x = Dropout(0.2)(x)
    output = Dense(output_neurons, output_activation)(t3)

    model = Model(inputs = input, outputs = output)

    return model
dim = (64,38,3)
output_neurons = 7
output_activation = 'softmax'
from keras import backend as Ke
Ke.clear_session()
model = RDLINet(dim, output_neurons, output_activation)

p=1;batch=256;
print('Random_seed_value== '+str(p))
X_train_1,X_test,Y_train_1,Y_test=train_test_split(X,Y,test_size=0.1,random_state=p)
X_train,X_val,Y_train,Y_val=train_test_split(X_train_1,Y_train_1,test_size=0.1,random_state=p)

print("================================================================================================")
opt =tf.keras.optimizers.Adam(learning_rate=0.008)
custom_early_stopping=tf.keras.callbacks.EarlyStopping(
monitor="val_accuracy",
patience=30,
verbose=1,
mode="auto",
baseline=0.92,
restore_best_weights=False,
start_from_epoch=170,
)
model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])
history = model.fit(X_train, Y_train, batch_size=128, epochs=300, verbose=1,validation_data=(X_val, Y_val),callbacks=[custom_early_stopping])

plt.figure(1)
plt.figure(figsize=(10,4))
plt.subplot(121)
plt.plot(history.history['loss'],'b--')
plt.plot(history.history['val_loss'],'g--')
plt.title('Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')

plt.subplot(122)
plt.plot(history.history['accuracy'],'b--')
plt.plot(history.history['val_accuracy'],'g--')
plt.title('Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='best')
plt.show()

results=model.evaluate(X_test,Y_test,batch_size=128,verbose=1)
print('Test loss:', results[0])
print('Test accuracy:', results[1])
predicted=model.predict(X_test,batch_size=128,verbose=1)
Y_pred=predicted.argmax(axis=-1)
Y_pred=predicted.argmax(axis=-1)
Y_predicted=pd.DataFrame(Y_test,
                          columns=['Asthma','Broncheostasis','Bronchiolitis',
                                  'COPD','Healthy','Pneumonia','URTI'])
Y_ori=[];Asthma_t=0;Broncheostasis_t=0;Bronchiolitis_t=0;
COPD_t=0;Healthy_t=0;Pneumonia_t=0;URTI_t=0;
for index,row in tqdm(Y_predicted.iterrows()):
    if row['Asthma']==1:
      Asthma_t=Asthma_t+1
      Y_ori.append(0)
    elif row['Broncheostasis']==1:
      Broncheostasis_t=Broncheostasis_t+1
      Y_ori.append(1)
    elif row['Bronchiolitis']==1:
      Bronchiolitis_t=Bronchiolitis_t+1
      Y_ori.append(2)
    elif row['COPD']==1:
      COPD_t=COPD_t+1
      Y_ori.append(3)
    elif row['Healthy']==1:
      Healthy_t=Healthy_t+1
      Y_ori.append(4)
    elif row['Pneumonia']==1:
      Pneumonia_t=Pneumonia_t+1
      Y_ori.append(5)
    elif row['URTI']==1:
      URTI_t=URTI_t+1
      Y_ori.append(6)

cm=confusion_matrix(Y_ori,Y_pred)
cm_norm=confusion_matrix(Y_ori,Y_pred,normalize='true')
print('Confusion Matrix');
print(cm)

cm_df = pd.DataFrame(cm,index = ['Asthma','Broncheostasis','Bronchiolitis','COPD','Healthy','Pneumonia','URTI'],
                      columns = ['Asthma','Broncheostasis','Bronchiolitis','COPD','Healthy','Pneumonia','URTI'])
plt.figure(100)
plt.figure(figsize=(20,6))
plt.subplot(121)
sns.heatmap(cm_df,annot=True,cmap="OrRd")
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')

cm_df_norm = pd.DataFrame(cm_norm, index = ['Asthma','Broncheostasis','Bronchiolitis','COPD','Healthy','Pneumonia','URTI'],
                          columns = ['Asthma','Broncheostasis','Bronchiolitis','COPD','Healthy','Pneumonia','URTI'])
plt.subplot(122)
sns.heatmap(cm_df_norm,annot=True,cmap="OrRd")
plt.title('Normalised Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

from tensorflow.keras.models import Model
out_dense= Model(inputs=model.input,outputs=model.get_layer('dense_2').output)
Y_denseout = out_dense.predict(X_test)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2,learning_rate='auto',init='random')
Y_embedded = tsne.fit_transform(Y_denseout)
Y_predicted=pd.DataFrame(Y_test,columns=['Asthma','Broncheostasis','Bronchiolitis','COPD','Healthy','Pneumonia','URTI'])
Y_ori=[];Asthma_t=0;Broncheostasis_t=0;Bronchiolitis_t=0;COPD_t=0;Healthy_t=0;Pneumonia_t=0;URTI_t=0;
for index,row in tqdm(Y_predicted.iterrows()):
   if row['Asthma']==1:
     Asthma_t=Asthma_t+1
     Y_ori.append(0)
   elif row['Broncheostasis']==1:
      Broncheostasis_t=Broncheostasis_t+1
      Y_ori.append(1)
   elif row['Bronchiolitis']==1:
      Bronchiolitis_t=Bronchiolitis_t+1
      Y_ori.append(2)
   elif row['COPD']==1:
       COPD_t=COPD_t+1
       Y_ori.append(3)
   elif row['Healthy']==1:
       Healthy_t=Healthy_t+1
       Y_ori.append(4)
   elif row['Pneumonia']==1:
       Pneumonia_t=Pneumonia_t+1
       Y_ori.append(5)
   elif row['URTI']==1:
      URTI_t=URTI_t+1
      Y_ori.append(6)


q=np.array(Y_ori)
label_l=[]
for i in range (X_test.shape[0]):
  if q[i]==0:
    label_l.append('Asthma')
  elif q[i]==1:
    label_l.append('Broncheostasis')
  elif q[i]==2:
    label_l.append('Bronchiolitis')
  elif q[i]==3:
    label_l.append('COPD')
  elif q[i]==4:
    label_l.append('Healthy')
  elif q[i]==5:
    label_l.append('Pneumonia')
  elif q[i]==6:
    label_l.append('URTI')

Label_ar=np.array(label_l)
df = pd.DataFrame()
df["Labels"] = Label_ar
df["comp-1"] = Y_embedded[:,0]
df["comp-2"] = Y_embedded[:,1]
import seaborn as sns
plt.figure(figsize=(8,6))
sns.scatterplot(x="comp-1", y="comp-2", hue=df.Labels.tolist(),
                data=df).set(title="T-SNE projection ")
predicted=model.predict(X_test,batch_size=128,verbose=1)
Y_pred=predicted.argmax(axis=-1)
prediction_output=[]
for i in range (Y_pred.shape[0]):
  if Y_pred[i]==0:
    prediction_output.append('Asthma')
  elif Y_pred[i]==1:
    prediction_output.append('Broncheostasis')
  elif Y_pred[i]==2:
    prediction_output.append('Bronchiolitis')
  elif Y_pred[i]==3:
    prediction_output.append('COPD')
  elif Y_pred[i]==4:
    prediction_output.append('Healthy')
  elif Y_pred[i]==5:
    prediction_output.append('Pneumonia')
  elif Y_pred[i]==6:
    prediction_output.append('URTI')

prediction_output_ar=np.array(prediction_output)
np.savetxt('prediction_output_ar_7cls.txt',prediction_output_ar,'%s')
from sklearn.metrics import classification_report
target_names = ['Asthma','Broncheostasis','Bronchiolitis','COPD','Healthy','Pneumonia','URTI']
print(classification_report(Y_ori, Y_pred, target_names=target_names))


from scipy import interp
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
Y_ori=np.array(Y_ori)
n_classes = 7 # number of class
# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(Y_ori, Y_softmax[:,i],pos_label=i )
    roc_auc[i] = auc(fpr[i], tpr[i])
p=[]
for i in range (7):
  p.append(roc_auc[i])
p=np.array(p)
p_rounded=np.round(p,decimals=3)
plt.figure(figsize=(5,4))
plt.plot(fpr[0], tpr[0], linestyle='--',color='m')
plt.plot(fpr[1], tpr[1], linestyle='--',color='orange')
plt.plot(fpr[2], tpr[2], linestyle='--',color='green')
plt.plot(fpr[3], tpr[3], linestyle='--',color='blue')
plt.plot(fpr[4], tpr[4], linestyle='--',color='red')
plt.plot(fpr[5], tpr[5], linestyle='--',color='pink')
plt.plot(fpr[6], tpr[6], linestyle='--',color='m')
plt.xlim([-0.005,0.05]); plt.grid()
plt.ylim([0.5,1.04])

# Plot the ROC chart
plt.figure(figsize=(9.5,5))
plt.plot(fpr[0], tpr[0], linestyle='--',color='m', label='ROC curve of Asthma class (area = '+str(p_rounded[0])+')')
plt.plot(fpr[1], tpr[1], linestyle='--',color='orange', label= 'ROC curve of  Broncheactasis class (area = '+str(p_rounded[1])+')')
plt.plot(fpr[2], tpr[2], linestyle='--',color='green', label='ROC curve of  Bronchiolitis class (area = '+str(p_rounded[2])+')')
plt.plot(fpr[3], tpr[3], linestyle='--',color='blue', label='ROC curve of  COPD class (area = '+str(p_rounded[3])+')')
plt.plot(fpr[4], tpr[4], linestyle='--',color='red', label='ROC curve of  Healthy class (area = '+str(p_rounded[4])+')')
plt.plot(fpr[5], tpr[5], linestyle='--',color='pink', label='ROC curve of  Pneumonia class (area = '+str(p_rounded[5])+')')
plt.plot(fpr[6], tpr[6], linestyle='--',color='m', label='ROC curve of fname=URTI class (area = '+str(p_rounded[6])+')')
plt.plot([0, 1], [0, 1], 'k--');plt.grid()
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
plt.show()
