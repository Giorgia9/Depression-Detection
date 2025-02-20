#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import os 
import glob
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import scikeras.wrappers as wrappers

import tensorflow as tf
from keras.models import Model

from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Dropout, Add, Multiply, TimeDistributed, BatchNormalization, Activation, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
import random
from tensorflow.keras.optimizers import AdamW


seed_value = 42
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)



# In[ ]:


# CREAZIONE DEI DATAFRAME 

condition_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/condition/*.csv')
control_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/control/*.csv')

# DATAFRAME per ogni time series dei pazienti affetti da depressione

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

    
# Parametri per le finestre
window_size =  2880 
stride = 2880  # passo di sovrapposizione

# Split basato sui pazienti 
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

X_train= X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # (num_samples, timesteps, features)
# Verifica dimensioni
print(f"Training set: {X_train.shape}, {y_train.shape}")
print(f"Validation set: {X_val.shape}, {y_val.shape}")


# In[ ]:


def wavenet_block(input_layer, filters_number, kernel_dim, dilatation, skip_filters=16):
    # Convoluzioni con gate e filtro 
    filter_output = Conv1D(filters=filters_number, kernel_size=kernel_dim, padding='causal', 
                           dilation_rate=dilatation, activation='tanh',
                           kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer) 
    gate_output = Conv1D(filters=filters_number, kernel_size=kernel_dim, padding='causal', 
                         dilation_rate=dilatation, activation='sigmoid',
                         kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_layer)
    
    output = Multiply()([filter_output, gate_output])
     
    # Residual connection
    residual = Conv1D(filters=filters_number, kernel_size=1, padding='same')(output)
    if input_layer.shape[-1] != filters_number:
        input_layer = Conv1D(filters=filters_number, kernel_size=1, padding='same')(input_layer)
    residual = Add()([residual, input_layer])  
    residual = BatchNormalization()(residual)  
    
    # Skip connection 
    skip = Conv1D(filters=skip_filters, kernel_size=1, padding='same')(output)
    return {"residual": residual, "skip": skip}

# Input layer
input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))

# Lista per accumulare le skip connections
skip_connection = []

# Definizione dei blocchi con numero variabile di filtri
residual = input_layer
filter_sizes = [32,32,64,64]  # Numero di filtri per ogni blocco
for i, dilation in enumerate([1, 2, 4,8]):
    block_output = wavenet_block(residual, filter_sizes[i], 3, dilation) 
    residual = block_output["residual"]
    skip_connection.append(block_output["skip"])
    residual = BatchNormalization()(residual)
    residual = Dropout(0.3)(residual)  # Regularization

# Somma di tutte le skip connections
skip_total = Add()(skip_connection) # prima c'era + [residual]
skip_total = Activation('relu')(skip_total)

# GlobalAveragePooling1D per ridurre la sequenza a un solo valore
output = GlobalAveragePooling1D()(skip_total)


# Output layer finale con sigmoid per la classificazione binaria
output_layer = Dense(1, activation="sigmoid")(output)

# Creazione del modello
model_3 = Model(inputs=input_layer, outputs=output_layer)

# Compilazione del modello con Adam
model_3.compile(optimizer=AdamW(learning_rate=0.0003, epsilon=1e-4), 
                loss='binary_crossentropy', metrics=['accuracy'])

y_train = np.array(y_train)
y_val = np.array(y_val)
print("Distribuzione nel training set:")
print(np.bincount(y_train))  # Conta le occorrenze di ogni classe

print("\nDistribuzione nel validation set:")
print(np.bincount(y_val))
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# ReduceLROnPlateau per ridurre il learning rate se la val_loss smette di migliorare
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)

class_weight = {0: 1., 1: 1.8}
# Addestra il modello
history = model_3.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100,callbacks=[early_stopping,lr_scheduler],batch_size=32,class_weight=class_weight)
    
# grafichiamo i valori della funzione di loss per ciascuna epoca 
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# nota: la funzione plot permette di disegnare una o più curve su una stessa area di disegno
# possiamo identificare ogni curva con l'etichetta label 

plt.plot(train_loss, label = 'training loss')
plt.plot(val_loss, label = 'validation loss')
plt.title('Loss values during training and validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# Mostra il grafico
plt.tight_layout()  # Migliora il layout  # Regola il layout
plt.show()
plt.close() 
# Valutare il modello
loss, accuracy = model_3.evaluate(X_val, y_val,batch_size=32)


# In[ ]:





# In[ ]:




