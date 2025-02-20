#!/usr/bin/env python
# coding: utf-8

# In[19]:


import pandas as pd 
import os
import glob 
from matplotlib import pyplot 
import seaborn as sns
import re
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np


# In[20]:


condition_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/condition/*.csv')
control_file_paths = glob.glob('/users/giorgia/Desktop/ML-programmi/depression/data/control/*.csv')


# In[48]:


# MODELLO che riceve in input time serie univariate (solo i dati di attività fisica)


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

# Parametri per le finestre     
window_size = 480  # 8 ore
stride = 480  # Nessuna sovrapposizione

# Split per pazienti (ogni paziente sarà o solo nel training o solo nel validation)
train_depressed, val_depressed = train_test_split(condition_df, test_size=0.3, random_state=42, shuffle=False)
train_control, val_control = train_test_split(control_df, test_size=0.3, random_state=42, shuffle=False)

# Funzione per generare le finestre con dimensione LSTM-friendly
def create_windows(patient_list, label, scaler=None):
    windows = []
    labels = []
    
    for df in patient_list:
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            window = df['activity'].iloc[start:end].values.reshape(-1, 1)  # Reshape per avere (steps, 1)
            
            # Standardizzazione finestra per finestra
            if scaler:
                window = scaler.transform(window)
            
            windows.append(window)
            labels.append(label)
            
    
    return np.array(windows), np.array(labels)

# Fit solo sui dati di training
scaler = StandardScaler()

# Estrazione e standardizzazione delle finestre
train_windows_depressed, train_labels_depressed = create_windows(train_depressed, label=1)
train_windows_control, train_labels_control = create_windows(train_control, label=0)

# Fit dello scaler solo su TRAIN
all_train_windows = np.vstack((train_windows_depressed.reshape(-1, 1), train_windows_control.reshape(-1, 1)))
scaler.fit(all_train_windows)

# Trasform su train e validation 
train_windows_depressed, _ = create_windows(train_depressed, label=1, scaler=scaler)
train_windows_control, _ = create_windows(train_control, label=0, scaler=scaler)

val_windows_depressed, val_labels_depressed = create_windows(val_depressed, label=1, scaler=scaler)
val_windows_control, val_labels_control = create_windows(val_control, label=0, scaler=scaler)

X_train = np.concatenate((train_windows_depressed, train_windows_control), axis=0)
y_train = np.concatenate((train_labels_depressed, train_labels_control), axis=0)

X_val = np.concatenate((val_windows_depressed, val_windows_control), axis=0)
y_val = np.concatenate((val_labels_depressed, val_labels_control), axis=0)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)

# addestramento del modello
model = xgb.XGBClassifier(objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,  # Più basso per migliorare generalizzazione
    max_depth=5,  # Evita overfitting
    min_child_weight=3,  # Evita split su poche istanze
    subsample=0.8,  # Uso solo il 70% dei dati per ogni albero
    colsample_bytree=0.9,  # Uso l'80% delle feature per ogni albero; 
    reg_lambda=0.7,  # Regularizzazione L2 aumentata
    reg_alpha=0.7,  # Regularizzazione L1 per ridurre feature irrilevanti
    n_estimators=300)  # Più alberi per maggiore capacità di apprendimento)
model.fit(X_train_flat, y_train)

# Previsioni sul set di validazione
y_pred = model.predict(X_val_flat)

# Calcolo dell'accuratezza
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.4f}')
    


# In[43]:


# MODELLO CHE CONSIDERA TIME SERIE MULTIVARIATE ( attività fisica + ora, minuti,giorno della settimana) 


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
    condition_df.append(df)
   
    
# DATAFRAME CONTROL GROUP - per ogni time series dei pazienti 'sani'- classe 0 (no depressione)

control_df = []
for file in control_file_paths:
    df = pd.read_csv(file)
    
    # considero solo una finestra temporale di 13 giorni per ogni time serie 
    df = df.drop(df.index[18720:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Converte in datetime
    
    # AGGIUNTE FEATURES TEMPORALI 
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)
    control_df.append(df)

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

# Parametri per le finestre
window_size = 2880  # 1 giorno
stride =  2880 # Nessuna sovrapposizione
# 240: 0.7670
# Split per pazienti (tutte le time serie di un paziente saranno o nel train set o nel test set )
train_depressed, val_depressed = train_test_split(condition_df, test_size=0.3, random_state=42, shuffle=False)
train_control, val_control = train_test_split(control_df, test_size=0.3, random_state=42, shuffle=False)

# Funzione per generare le finestre con dimensione LSTM-friendly
def create_windows(patient_list,columns_to_use,label, scaler=None):
    windows = []
    labels = []
    
    for df in patient_list:
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            window = df[columns_to_use].iloc[start:end].values.reshape(-1, 5)
            
            # Standardizzazione finestra per finestra
            if scaler:
                window = scaler.transform(window)
            
            windows.append(window)
            labels.append(label)
    
    return np.array(windows), np.array(labels)
    
columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos']


# Fit dello scaler solo sui dati di training
scaler = StandardScaler()

# Creazione e standardizzazione delle finestre
train_windows_depressed, train_labels_depressed = create_windows(train_depressed,columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos'],label=1)
train_windows_control, train_labels_control = create_windows(train_control,columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos'], label=0)

# Fit dello scaler solo su TRAIN
all_train_windows = np.vstack((train_windows_depressed.reshape(-1, 5), train_windows_control.reshape(-1, 5)))
scaler.fit(all_train_windows)

# scalar.transform sulle finestre
train_windows_depressed, _ = create_windows(train_depressed,columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos'],label=1, scaler=scaler)
train_windows_control, _ = create_windows(train_control,columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos'], label=0, scaler=scaler)

val_windows_depressed, val_labels_depressed = create_windows(val_depressed,columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos'], label=1, scaler=scaler)
val_windows_control, val_labels_control = create_windows(val_control,columns_to_use = ['activity', 'hour', 'minute', 'day_sin','day_cos'],label=0, scaler=scaler)


# Concatenazione dei dati
X_train = np.concatenate((train_windows_depressed, train_windows_control), axis=0)
y_train = np.concatenate((train_labels_depressed, train_labels_control), axis=0)

X_val = np.concatenate((val_windows_depressed, val_windows_control), axis=0)
y_val = np.concatenate((val_labels_depressed, val_labels_control), axis=0)

X_train = X_train.reshape(X_train.shape[0], -1)
X_val  = X_val.reshape(X_val.shape[0], -1)

model = xgb.XGBClassifier(objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,  # Più basso per migliorare generalizzazione
    max_depth=5,  # Evita overfitting
    min_child_weight=3,  # Evita split su poche istanze
    subsample=0.8,  # Uso solo il 70% dei dati per ogni albero
    colsample_bytree=0.9,  # Uso l'80% delle feature per ogni albero; 
    reg_lambda=0.7,  # Regularizzazione L2 aumentata
    reg_alpha=0.7,  # Regularizzazione L1 per ridurre feature irrilevanti
    n_estimators=300 ) # Più alberi per maggiore capacità di apprendimento)
model.fit(X_train, y_train)

# Previsioni sul set di validazione
y_pred = model.predict(X_val)

# Calcolo dell'accuratezza
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.4f}')


# In[36]:


# MODELLO CHE CONSIDERA TIME SERIE MULTIVARIATE + dati statistici sull'attività motoria

# DATAFRAME per ogni time series dei pazienti affetti da depressione
# e visualizzazione 

condition_df = []  # lista dei dataframes
for file in condition_file_paths:
    df = pd.read_csv(file)
     # Estrai il nome del file (senza percorso)
    
    # considero solo una finestra temporale di 13 giorni (18720 min) per ogni time serie 
    df = df.drop(df.index[18720:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Converte in datetime
    
    
    # AGGIUNTE FEATURES TEMPORALI 
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df = df.drop(columns=['timestamp'])
    condition_df.append(df)
   
    
# DATAFRAME CONTROL GROUP - per ogni time series dei pazienti 'sani'- classe 0 (no depressione)

control_df = []
for file in control_file_paths:
    df = pd.read_csv(file)
    # considero solo una finestra temporale di 13 giorni per ogni time serie 
    df = df.drop(df.index[18720:])
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # Converte in datetime
    
    # AGGIUNTE FEATURES TEMPORALI 
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.weekday / 7)
    df = df.drop(columns=['timestamp'])
    control_df.append(df)


# Parametri per le finestre
window_size = 240  # 4 ore
stride = 240  # Nessuna sovrapposizione

# Split per pazienti (tutte le time serie di un paziente saranno o nel train set o nel test set )
train_depressed, val_depressed = train_test_split(condition_df, test_size=0.3, random_state=42, shuffle=False)
train_control, val_control = train_test_split(control_df, test_size=0.3, random_state=42, shuffle=False)

def create_windows(patient_list, columns_to_use, label):
    windows = []
    labels = []
    
    for df in patient_list:
        for start in range(0, len(df) - window_size + 1, stride):
            end = start + window_size
            window = df[columns_to_use].iloc[start:end].values  # (480, 5)

            # Estrazione della colonna "activity" (prima colonna)
            activity_values = window[:, 0]  # (480,)
            
            # Calcolo statistiche solo su activity
            mean_activity = np.mean(activity_values)  
            std_activity = np.std(activity_values)    
            sum_activity = np.sum(activity_values)    
            max_activity = np.max(activity_values)    

            # Appiattimento della finestra e aggiunta delle statistiche
            flat_window = window.flatten()  # (480*5,)
            stats_features = np.array([mean_activity, std_activity, sum_activity, max_activity])  # (4,)            

            full_window = np.concatenate([flat_window, stats_features])  # (480*5 + 4,)
            
            windows.append(full_window)
            labels.append(label)
    
    return np.array(windows), np.array(labels)

    
columns_to_use = ['activity', 'hour','minute', 'day_sin','day_cos']


# Creazione finestre per train e validation
train_windows_depressed, train_labels_depressed = create_windows(train_depressed, columns_to_use, label=1)
train_windows_control, train_labels_control = create_windows(train_control, columns_to_use, label=0)
val_windows_depressed, val_labels_depressed = create_windows(val_depressed, columns_to_use, label=1)
val_windows_control, val_labels_control = create_windows(val_control, columns_to_use, label=0)

# Concatenazione delle finestre
X_train = np.concatenate((train_windows_depressed, train_windows_control), axis=0)
y_train = np.concatenate((train_labels_depressed, train_labels_control), axis=0)
X_val = np.concatenate((val_windows_depressed, val_windows_control), axis=0)
y_val = np.concatenate((val_labels_depressed, val_labels_control), axis=0)

# Standardizzazione solo sulle prime 5 feature (escludendo le statistiche di activity)
scaler = StandardScaler()
scaler.fit(X_train[:, :-4])  # Fit solo sulle feature originali, escludendo le 4 statistiche statistiche

# Funzione per applicare lo scaler mantenendo inalterate le statistiche
def scale_windows(X, scaler):
    X_scaled = X.copy()
    X_scaled[:, :-4] = scaler.transform(X[:, :-4])  # Standardizziamo solo le prime 5 feature
    return X_scaled

# Standardizzazione delle finestre
X_train = scale_windows(X_train, scaler)
X_val = scale_windows(X_val, scaler)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,  # Più basso per migliorare generalizzazione
    max_depth=5,  # Evita overfitting
    min_child_weight=3,  # Evita split su poche istanze
    subsample=0.8,  # Uso solo il 70% dei dati per ogni albero
    colsample_bytree=0.9,  # Uso l'80% delle feature per ogni albero; 
    reg_lambda=0.7,  # Regularizzazione L2 aumentata
    reg_alpha=0.7,  # Regularizzazione L1 per ridurre feature irrilevanti
    n_estimators=300  # Più alberi per maggiore capacità di apprendimento
)
model.fit(X_train, y_train)

# Previsioni sul set di validazione
y_pred = model.predict(X_val)

# Calcolo dell'accuratezza
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy:.4f}')


# In[ ]:





# In[ ]:




