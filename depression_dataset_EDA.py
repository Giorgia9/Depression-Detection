#!/usr/bin/env python
# coding: utf-8

#  STATISTICAL SUMMARY: Analisi dei dati grezzi  

# In[1]:


import pandas as pd 
import os
import glob 
from matplotlib import pyplot 
import seaborn as sns


# In[4]:


madrs_scores_path = '/users/giorgia/Desktop/ML-programmi/depression/data/scores.csv'
condition_path ='/users/giorgia/Desktop/ML-programmi/depression/data/condition'

madrs_df = pd.read_csv(madrs_scores_path)
condition_madrs_df = madrs_df.head(23)

# size MADRS - dataset
print('DIMENSIONE madrs-dataset: ', madrs_df.shape)

# NOTE
# Nan indica che il paziente non è affetto da depressione 

# classe 1: bipolare II ( fasi depressive alternata a episodi ipo-maniacali)
# un episodio ipo-maniacale indica un umore alterato e persistentemente alto per 
# almeno 4 giorni; in questa fase diminuisce il bisogno di dormire (dopo 3 ore di sonno 
# ci si sente riposati); gli episodi ipo-maniacali sono meno gravi di quelli maniacali

# classe 2: unipolare con disturbo depressivo 

# classe 3: bipolare I (fasi depressive alternate a episodi maniacali)
# episodio maniacale: umore estremamente elevato continuativamente per almeno 1 settimana 

print(condition_madrs_df)
print('\nSTATISTICHE')
print(condition_madrs_df.describe())

# valore medio afftype:  0.727273 -> sbilanciamento verso la classe 2 

print('\nISTOGRAMMI')
condition_madrs_df.hist()
pyplot.show()

# distribuzione dei valori per classe 
print("CLASS DISTRIBUTION\nafftype 1 : Bipolar II\nafftype 2: unipolar depressed\nafftype 3:bipolar I\n")
print(condition_madrs_df.groupby('afftype').size())


# In[9]:


print("Numero di pazienti per valore di madrs1")
condition_madrs_df['madrs1'].value_counts()


# depressione lieve (madrs nell'intervallo [7-19]) : 7 pazienti \n
# depressione moderata (madrs nell'intervallo [19-34]): 16 pazienti \n
# 
# Nel dataset non sono riportati esempi di pazienti con depressione severa. Le classi non sono distribuite in modo equo per poter 
# effettuare una classificazione multi-classe in base ai valori di madrs
# 
# 

# In[10]:


# matrice di correlazione 
print(" Matrice di correlazione ") 
sns.heatmap(condition_madrs_df.corr(numeric_only=True), annot=True)
    


# Riportiamo di seguito le correlazioni tra features che sono risultate più rilevanti 
# 
# madrs1 - madrs2 : 0.65 corr. positiva \n  
# madrs1 - inpatient : -0.52  corr.negativa (dovuta a come è stato codificato il valore inpatient) \n 
# madrs2 - inpatient : -0.41 corr. negativa (dovuta a come è stato codificato il valore inpatient) \n
# melancholia - inpatient : 0.4  (corr. positiva) \n 
# marriage - madrs1 : 0.45 (corr. positiva) \n 
# 
# Di seguito sono state analizzate alcune di queste correlazioni attraverso dei grafici a barre 
# 

# In[11]:


print("Relazione tra inpatient e madrs1  - grafico a barre ")
sns.boxplot(x="inpatient", y="madrs1", data=condition_madrs_df)


# RELAZIONE melancholia - inpatient 
# il grafico a barre mostra che:
# quando inpatient = 1 (paziente ricoverato), i valori di madrs1 rappresentano una depressione moderata (26-29)
# quando inpatient = 2 (paziente non ricoverato), i valori di madrs1 rappresentano una depressione lieve (18-25)
# 
# Quindi è più probabile che un paziente ricoverato abbia un valore di depressione maggiore 
# Il grafico conferma il valore di correlazione inpatient-madrs1 = -0.52
# 
# 

# In[13]:


print("Relazione tra marriage e madrs1  - grafico a barre ")
sns.boxplot(x="marriage", y="madrs1", data=condition_madrs_df)


# In[ ]:





# RELAZIONE marriage - madrs1 
# Tra i pazienti depressi non sposati (marriage = 2), i valori di madrs sono tendenzialmente alti: 25-29 con qualche valore anomalo
# Tra i pazienti depressi sposati o che convivono (marriage = 1) i valori di madrs variano in un intervallo più ampio: 13-28

# In[12]:


print("Relazione tra madrs1 e madrs2  - grafico a barre ")
sns.boxplot(x="madrs1", y="madrs2", data=condition_madrs_df)


# RELAZIONE madrs1 - madrs2 
# Correlazione : madrs1 - madrs2 : 0.65 (man mano che consideriamo valori di madrs1 più elevati, il rispettivo madrs2 aumenta) \n  
# 
# Dai grafici a baffi si evince che: 
# Tendenzialmente madrs2 < madrs1 , ossia i valori di madrs tendono a essere minori a fine misurazione.
# Per pochi casi madrs1 = madrs2 oppure madrs2 > madrs1. 
# Nonostante i cambiamenti dei valori di madrs, ogni paziente rimane nella propria fascia depressiva 
