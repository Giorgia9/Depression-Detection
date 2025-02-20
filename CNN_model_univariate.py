#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import os 
import glob
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

import scikeras.wrappers as wrappers

import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, AveragePooling1D, Dropout, Add, MaxPooling1D, ReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.preprocessing import Normalizer
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization

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
control_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/control/*.csv')


# DATAFRAME per ogni time series dei pazienti affetti da depressione
# e visualizzazione 

condition_df = []  # lista dei dataframes
for file in condition_file_paths:
    df = pd.read_csv(file)
    
    # considero solo una finestra temporale di 13 giorni (18720 min) per ogni time serie 
    df = df.drop(df.index[18720:])
    df.drop(['timestamp'],axis=1)
    condition_df.append(df)

    
# DATAFRAME CONTROL GROUP - per ogni time series dei pazienti 'sani'- classe 0 (no depressione)

control_df = []
for file in control_file_paths:
    df = pd.read_csv(file)
    
    # considero solo una finestra temporale di 13 giorni per ogni time serie 
    df = df.drop(df.index[18720:])
    df.drop(['timestamp'],axis=1)
    control_df.append(df)

    


# In[ ]:


# giorni di misurazione per paziente depresso 

xticks =[]
labels =[]
for i in range(23):
    xticks.append(i)
    labels.append('c'+str(i+1))
    
plt.scatter(madrs_df['number'][:23],madrs_df['days'][:23])
# xticks indica dove devono essere posizionate le etichette sull'asse x 
plt.xticks(xticks,labels=labels)
print('\nGiorni di misurazione dello stato depressivo dei pazienti')
print('\nPaziente con numero di giorni min: condition8 - 5 days')
print('\nPaziente con numero di giorni max: condition2 - 18 giorni')
plt.show()

# giorni di misurazione per paziente non depresso 

xticks =[]
labels =[]
for i in range(32):
    xticks.append(i)
    labels.append('c'+str(i+1))
    
plt.scatter(madrs_df['number'][23:],madrs_df['days'][23:])
# xticks indica dove devono essere posizionate le etichette sull'asse x 
plt.xticks(xticks,labels=labels)
print('\nGiorni di misurazione dei pazienti non depressi')
plt.show()



# In[14]:


from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# Parametri per le finestre
window_size = 2880  # 1 giorno
stride = 1440  # passo di sovrapposizione

# Split basato sui pazienti (separiamo prima depressi e sani!)
train_depressed, val_depressed = train_test_split(condition_df, test_size=0.3, random_state=42)
train_control, val_control = train_test_split(control_df, test_size=0.3, random_state=42)

# Funzione per generare le finestre
def create_windows(patient_list, label):
    windows = []
    labels = []
    for df in patient_list:
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            window = df['activity'].iloc[start:end].values
            windows.append(window)
            labels.append(label)
    return windows, labels

# Genera le finestre per TRAIN
train_windows_depressed, train_labels_depressed = create_windows(train_depressed, label=1)
train_windows_control, train_labels_control = create_windows(train_control, label=0)

# Genera le finestre per VALIDATION
val_windows_depressed, val_labels_depressed = create_windows(val_depressed, label=1)
val_windows_control, val_labels_control = create_windows(val_control, label=0)

# Concateniamo i dati
X_train = np.array(train_windows_depressed + train_windows_control)
y_train = np.array(train_labels_depressed + train_labels_control)

X_val = np.array(val_windows_depressed + val_windows_control)
y_val = np.array(val_labels_depressed + val_labels_control)

# Standardizzazione SOLO sui dati di training
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit + transform sul training
X_val = scaler.transform(X_val)  # Solo transform sul validation set

# Verifica dimensioni
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")


# In[7]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Creazione del modello
model_2 = Sequential()

# Input layer
model_2.add(Input(shape=(window_size, 1)))

model_2.add(Conv1D(filters=16, kernel_size=5, strides=2, padding='valid'))
model_2.add(BatchNormalization())  
model_2.add(ReLU())
model_2.add(Dropout(0.3))
model_2.add(MaxPooling1D(pool_size=2, strides=2))

# BLOCCO II - Seconda convoluzione
model_2.add(Conv1D(filters=32, kernel_size=3, strides=1, padding='same'))
model_2.add(BatchNormalization())
model_2.add(ReLU())
model_2.add(Dropout(0.4))
model_2.add(MaxPooling1D(pool_size=2, strides=2))


# Appiattimento del risultato e layer fully connected
model_2.add(Flatten())
model_2.add(Dense(16,kernel_regularizer=tf.keras.regularizers.l2(0.001)))

# Output layer
model_2.add(Dense(1, activation='sigmoid'))  # Classificazione binaria

# Compilazione del modello
model_2.compile(optimizer=Adam(learning_rate=0.0005), 
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


# In[ ]:




