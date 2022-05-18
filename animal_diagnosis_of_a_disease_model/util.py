import os
import cv2
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

X = []
Y = []

def cm_train_data(filename,dir):
#     data_dir = 'hackaton_dataset'
    CM_ABN = [f for f in os.listdir(data_dir+'/CM_ABN') if f.endswith('.jpg')]
    CM_NOR = [f for f in os.listdir(data_dir+'/CM_NOR') if f.endswith('.jpg')]
    
    for img in CM_ABN:
        if "Ab01" in img:
            X.append(img_to_array(load_img(data_dir+'/CM_ABN/'+img)).flatten() / 255.0)
            Y.append(1)
        elif "Ab04" in img:
            X.append(img_to_array(load_img(data_dir+'/CM_ABN/'+img)).flatten() / 255.0)
            Y.append(2)
        elif "Ch02" in img:
            X.append(img_to_array(load_img(data_dir+'/CM_ABN/'+img)).flatten() / 255.0)
            Y.append(3)
        elif "Ch05" in img:
            X.append(img_to_array(load_img(data_dir+'/CM_ABN/'+img)).flatten() / 255.0)
            Y.append(4)
        elif "Mu02" in img:
            X.append(img_to_array(load_img(data_dir+'/CM_ABN/'+img)).flatten() / 255.0)
            Y.append(5)
        elif "Mu04" in img:
            X.append(img_to_array(load_img(data_dir+'/CM_ABN/'+img)).flatten() / 255.0)
            Y.append(6)
    for img in CM_NOR:
        X.append(img_to_array(load_img(data_dir+'/CM_NOR/'+img)).flatten() / 255.0)
        Y.append(0)
        
    Y_val_org = Y

    #Normalization
    X = np.array(X)
    Y = tf.keras.utils.to_categorical(Y)

    #Reshape
    X = X.reshape(-1, 128, 128, 3)

    #Train-Test split
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)
            
    return X_train,Y_train
            
        
def if_train_data(filename,dir):
#     data_dir = 'hackaton_dataset'
    IF_ABN = [f for f in os.listdir(data_dir+'/IF_ABN') if f.endswith('.jpg')]
    IF_NOR = [f for f in os.listdir(data_dir+'/IF_NOR') if f.endswith('.jpg')]
    
    for img in IF_ABN:
        if "Ab01" in img:
            X.append(img_to_array(load_img(data_dir+'/IF_ABN/'+img)).flatten() / 255.0)
            Y.append(1)
        elif "Ab04" in img:
            X.append(img_to_array(load_img(data_dir+'/IF_ABN/'+img)).flatten() / 255.0)
            Y.append(2)
        elif "Ch02" in img:
            X.append(img_to_array(load_img(data_dir+'/IF_ABN/'+img)).flatten() / 255.0)
            Y.append(3)
        elif "Ch05" in img:
            X.append(img_to_array(load_img(data_dir+'/IF_ABN/'+img)).flatten() / 255.0)
            Y.append(4)
        elif "Mu02" in img:
            X.append(img_to_array(load_img(data_dir+'/IF_ABN/'+img)).flatten() / 255.0)
            Y.append(5)
        elif "Mu04" in img:
            X.append(img_to_array(load_img(data_dir+'/IF_ABN/'+img)).flatten() / 255.0)
            Y.append(6)
    for img in IF_NOR:
        X.append(img_to_array(load_img(data_dir+'/IF_NOR/'+img)).flatten() / 255.0)
        Y.append(0)    
        
    Y_val_org = Y

    #Normalization
    X = np.array(X)
    Y = tf.keras.utils.to_categorical(Y)
        
    return X_train,Y_train

def im_train_data(filename,dir):
#     data_dir = 'hackaton_dataset'
    IM_ABN = [f for f in os.listdir(data_dir+'/IM_ABN') if f.endswith('.jpg')]
    IM_NOR = [f for f in os.listdir(data_dir+'/IM_NOR') if f.endswith('.jpg')]
    
    for img in IM_ABN:
        if "Ab01" in img:
            X.append(img_to_array(load_img(data_dir+'/IM_ABN/'+img)).flatten() / 255.0)
            Y.append(1)
        elif "Ab04" in img:
            X.append(img_to_array(load_img(data_dir+'/IM_ABN/'+img)).flatten() / 255.0)
            Y.append(2)
        elif "Ch02" in img:
            X.append(img_to_array(load_img(data_dir+'/IM_ABN/'+img)).flatten() / 255.0)
            Y.append(3)
        elif "Ch05" in img:
            X.append(img_to_array(load_img(data_dir+'/IM_ABN/'+img)).flatten() / 255.0)
            Y.append(4)
        elif "Mu02" in img:
            X.append(img_to_array(load_img(data_dir+'/IM_ABN/'+img)).flatten() / 255.0)
            Y.append(5)
        elif "Mu04" in img:
            X.append(img_to_array(load_img(data_dir+'/IM_ABN/'+img)).flatten() / 255.0)
            Y.append(6)
    for img in IM_NOR:
        X.append(img_to_array(load_img(data_dir+'/IM_NOR/'+img)).flatten() / 255.0)
        Y.append(0)   
        
    Y_val_org = Y

    #Normalization
    X = np.array(X)
    Y = tf.keras.utils.to_categorical(Y) 
       
    return X_train,Y_train
        
def animal_diagnosis_of_a_disease_model(x,y,d):
    input_shape = (x,y,d)

    from tensorflow.keras.applications import InceptionResNetV2
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.layers import InputLayer
    from tensorflow.keras.layers import GlobalAveragePooling2D
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.models import Model
    from tensorflow.keras import optimizers
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    googleNet_model = InceptionResNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    googleNet_model.trainable = True
    model = Sequential()
    model.add(googleNet_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(units=7, activation='softmax'))
    model.compile(loss='cross_crossentropy',
                  optimizer=optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  metrics=['accuracy'])
    model.summary()
    
    return model


def train_model(model, X, Y, BATCH_SIZE, EPOCHS):
    #Currently not used
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=2,
                                   verbose=0, mode='auto')
    history = model.fit(X_train, Y_train, batch_size = BATCH_SIZE, epochs = EPOCHS, validation_data = (X_val, Y_val), verbose = 1)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 4))
    t = f.suptitle('Pre-trained InceptionResNetV2 Transfer Learn with Fine-Tuning & Image Augmentation Performance ', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,EPOCHS+1))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, EPOCHS+1, 1))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch #')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, EPOCHS+1, 1))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch #')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
        
    return history    
        
def im_save_model(model, filename = "im_model"):
    """
    model 저장 함수
    
    Argument:
    model -- 저장할 모델
    filename -- 저장할 파일 이름
    
    Returns:
    None
    """
    
    # 훈련된 모델과 파라미터를 저장할 폴더 생성
    os.mkdir(filename)
    
    model.save(filename + "/" + filename + ".h5")        

def if_save_model(model, filename = "if_model"):
    """
    model 저장 함수
    
    Argument:
    model -- 저장할 모델
    filename -- 저장할 파일 이름
    
    Returns:
    None
    """
    
    # 훈련된 모델과 파라미터를 저장할 폴더 생성
    os.mkdir(filename)
    
    model.save(filename + "/" + filename + ".h5")   
    
def cm_save_model(model, filename = "cm_model"):
    """
    model 저장 함수
    
    Argument:
    model -- 저장할 모델
    filename -- 저장할 파일 이름
    
    Returns:
    None
    """
    
    # 훈련된 모델과 파라미터를 저장할 폴더 생성
    os.mkdir(filename)
    
    model.save(filename + "/" + filename + ".h5")   
        
def im_load_trained_model(filename = "im_model"):
    """
    훈련된 model을 불러오는 함수
    
    Argument:
    filename -- 저장할 파일 이름
    """
    
    model = load_model(filename + "/" + filename + ".h5")
    
    return model 

def cm_load_trained_model(filename = "cm_model"):
    """
    훈련된 model을 불러오는 함수
    
    Argument:
    filename -- 저장할 파일 이름
    """
    
    model = load_model(filename + "/" + filename + ".h5")
    
    return model 

def if_load_trained_model(filename = "if_model"):
    """
    훈련된 model을 불러오는 함수
    
    Argument:
    filename -- 저장할 파일 이름
    """
    
    model = load_model(filename + "/" + filename + ".h5")
    
    return model 
        
def inference(model, X_test):
    """
    훈련된 model을 사용하여 입력 데이터 X의 label과 confience를 추정하는 함수
    
    Argument:
    model -- 훈련된 keras 모델
    X -- 입력 데이터
    
    Returns:
    labels -- 입력 데이터에 대한 예측된 label 값
    confidence -- label에 대한 확률 값
    """
    
    confidences = model.predict(X_test)
    labels = np.round(confidences)
    
    return labels, confidences        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        