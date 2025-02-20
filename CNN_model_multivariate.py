#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import os 
import glob
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, AveragePooling1D, Dropout, Add, MaxPooling1D, ReLU,BatchNormalization,GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import Normalizer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

import random
from keras.regularizers import l2


# In[2]:


seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)


# In[3]:


# CREAZIONE DEI DATAFRAME per i file condition.csv e scores.csv
# restituisce una lista con i path-names che rispettano l'espressione regolare passata 
# come parametro

condition_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/condition/*.csv')
condition_file_paths.sort()
control_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/control/*.csv')

# DATAFRAME per ogni time series dei pazienti affetti da depressione
# e visualizzazione 

condition_df = []  # lista dei dataframes
for file in condition_file_paths:
    df = pd.read_csv(file)
    
    # considero solo una finestra temporale di 13 giorni (18720 min) per ogni time serie 
    df = df.drop(df.index[18720:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Converte in datetime
    
    
    # AGGIUNTE FEATURES TEMPORALI 
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df = df.drop(['timestamp'],axis=1)
   
    condition_df.append(df)
   
# DATAFRAME CONTROL GROUP - per ogni time series dei pazienti 'sani'- classe 0 (no depressione)

control_df = []
for file in control_file_paths:
    df = pd.read_csv(file)
    
    # considero solo una finestra temporale di 13 giorni per ogni time serie 
    df = df.drop(df.index[18720:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Converte in datetime
    
    
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df = df.drop(['timestamp'],axis=1)
 
    control_df.append(df)

    


# In[8]:


# Parametri per le finestre
window_size = 2880  # 2 giorni
stride = 720  # passo di sovrapposizione

# Split basato sui pazienti 
train_depressed, val_depressed = train_test_split(condition_df, test_size=0.3, random_state=42)
train_control, val_control = train_test_split(control_df, test_size=0.3, random_state=42)

# Funzione per generare le finestre
def create_windows(patient_list, label,columns_to_use):
    windows = []
    labels = []
    for df in patient_list:
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            # Selezioniamo tutte le features tranne `timestamp`
            window = df[columns_to_use].iloc[start:end].values 
            windows.append(window)
            labels.append(label)
    
    return windows, labels

columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos']

# Genera le finestre per TRAIN
train_windows_depressed, train_labels_depressed = create_windows(train_depressed, label=1, columns_to_use=['activity', 'hour', 'minute','day_sin','day_cos' ])
train_windows_control, train_labels_control = create_windows(train_control, label=0,columns_to_use=['activity', 'hour', 'minute','day_sin','day_cos'])

# Genera le finestre per VALIDATION
val_windows_depressed, val_labels_depressed = create_windows(val_depressed, label=1,columns_to_use=['activity', 'hour', 'minute','day_sin','day_cos'])
val_windows_control, val_labels_control = create_windows(val_control, label=0,columns_to_use=['activity', 'hour', 'minute','day_sin','day_cos'])

X_train = np.array(train_windows_depressed + train_windows_control)
y_train = np.array(train_labels_depressed + train_labels_control)

X_val = np.array(val_windows_depressed + val_windows_control)
y_val = np.array(val_labels_depressed + val_labels_control)


#  l'indice della colonna 'dayofweek'
day_sin_index = list(columns_to_use).index('day_sin')
day_cos_index = list(columns_to_use).index('day_cos')


#  StandardScaler per ogni feature numerica
scalers = [StandardScaler() for _ in range(len(columns_to_use) - 1)]

# Standardizzazione di tutte le features tranne 'dayofweek'
for i in range(len(columns_to_use)):
    if i != day_sin_index and i!= day_cos_index:  # Escludiamo il giorno della settimana
        X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])
        X_val[:, :, i] = scalers[i].transform(X_val[:, :, i])


# Verifica dimensioni
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")


# In[9]:


# Creazione del modello
model_2 = Sequential()

# Input layer
model_2.add(Input(shape=(window_size, 5)))

# I BLOCCO 
model_2.add(Conv1D(filters=16, kernel_size=5, strides=1,padding='valid')) 
model_2.add(BatchNormalization())  
model_2.add(ReLU())

# II BLOCCO 
model_2.add(Conv1D(filters=32, kernel_size=5, strides=2,padding='valid'))
model_2.add(BatchNormalization())  
model_2.add(ReLU())

# III BLOCCO 
model_2.add(Conv1D(filters=64, kernel_size=5, strides=2,padding='valid',kernel_regularizer=tf.keras.regularizers.l2(0.0001))) 
model_2.add(BatchNormalization())  
model_2.add(ReLU())
model_2.add(MaxPooling1D(pool_size=2, strides=2))


# BLOCCO IV - Seconda convoluzione
model_2.add(Conv1D(filters=64, kernel_size=5, strides=1, dilation_rate=2,padding='valid',kernel_regularizer=tf.keras.regularizers.l2(0.0001)))
model_2.add(BatchNormalization())
model_2.add(ReLU())
model_2.add(AveragePooling1D(pool_size=2, strides=2))


# Appiattimento del risultato e layer fully connected
model_2.add(Flatten())
model_2.add(Dense(64,kernel_regularizer=tf.keras.regularizers.l2(0.01)))
model_2.add(Dropout(0.4))
model_2.add(Dense(32))
model_2.add(Dropout(0.5))


# Output layer
model_2.add(Dense(1, activation='sigmoid'))  # Classificazione binaria

# Compilazione del modello
model_2.compile(optimizer=Adam(learning_rate=0.0003,epsilon=1e-7), 
                loss='binary_crossentropy', 
                metrics=['accuracy'])


# Callback per Early Stopping e riduzione del learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
#lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=20, min_lr=1e-6, verbose=1)


# Addestramento del modello
history = model_2.fit(X_train, y_train, validation_data=(X_val, y_val), 
                      callbacks=[early_stopping], epochs=100,batch_size=32)

# Grafico della loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss values during training and validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Valutazione del modello
loss, accuracy = model_2.evaluate(X_val, y_val,batch_size=32)
print(f"Validation Accuracy: {accuracy:.4f}")


# In[11]:


# FULLY CONVOLUTIONAL NEURAL NETWORK FCN 
# Creazione del modello
model_2 = Sequential()

# Input layer
model_2.add(Input(shape=(window_size, 5)))


# I BLOCCO 
model_2.add(Conv1D(filters=128, kernel_size=8, strides=1,padding='valid'))
model_2.add(BatchNormalization())  
model_2.add(ReLU())

# II BLOCCO 
model_2.add(Conv1D(filters=256, kernel_size=5, strides=1,padding='valid'))
model_2.add(BatchNormalization())  
model_2.add(ReLU())

# III BLOCCO 
model_2.add(Conv1D(filters=128, kernel_size=3, strides=1,padding='valid') )
model_2.add(BatchNormalization())  
model_2.add(ReLU())
model_2.add(GlobalAveragePooling1D())

# Output layer
model_2.add(Dense(1, activation='sigmoid'))  # Classificazione binaria

# Compilazione del modello
model_2.compile(optimizer=Adam(learning_rate=0.0003,epsilon=1e-7), 
                loss='binary_crossentropy', 
                metrics=['accuracy'])


# Callback per Early Stopping e riduzione del learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Addestramento del modello
history = model_2.fit(X_train, y_train, validation_data=(X_val, y_val), 
                      callbacks=[early_stopping], epochs=20,batch_size=32)

# Grafico della loss
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Loss values during training and validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Valutazione del modello
loss, accuracy = model_2.evaluate(X_val, y_val,batch_size=32)
print(f"Validation Accuracy: {accuracy:.4f}")


# In[ ]:





# In[ ]:




