#!/usr/bin/env python
# coding: utf-8

# In[4]:


#Φέρνουμε τη βιβλιοθήκη numpy και την ονομάζουμε np
import numpy as np

#Φέρνουμε τη βιβλιοθήκη pandas και την ονομάζουμε pd
import pandas as pd


# In[5]:


#Φορτώνουμε τα δεδομένα του αρχείου Fake.csv στη μεταβλητή "fake"
fake = pd.read_csv("Fake.csv")

#Μετράμε πόσες φορές εμφανίζονται οι "κρίσιμες λέξεις" και προσθέτουμε τις αντίστοιχες στήλες
fake["to"] = fake['title'].str.count("to ")
fake["in"] = fake['title'].str.count("in ")
fake["on"] = fake['title'].str.count("on ")
fake["On"] = fake['title'].str.count("On ")
fake["with"] = fake['title'].str.count("with ")
fake["THE"] = fake['title'].str.count("THE ")

#Προσθέτουμε τις στήλες για τις κλάσεις
fake['target'] = 0
fake['Validity'] = "fake"

#Εμφανίζουμε τις πέντε πρώτες γραμμές των δεδομένων μας
fake.head()


# In[6]:


#Φορτώνουμε τα δεδομένα του αρχείου True.csv στη μεταβλητή "true"
true = pd.read_csv("True.csv")

#Μετράμε πόσες φορές εμφανίζονται οι "κρίσιμες λέξεις" και προσθέτουμε τις αντίστοιχες στήλες
true["to"] = true['title'].str.count("to ")
true["in"] = true['title'].str.count("in ")
true["on"] = true['title'].str.count("on ")
true["On"] = true['title'].str.count("On ")
true["with"] = true['title'].str.count("with ")
true["THE"] = true['title'].str.count("THE ")

#Προσθέτουμε τις στήλες για τις κλάσεις
true['target'] = 1
true['Validity'] = "true"

#Εμφανίζουμε τις πέντε πρώτες γραμμές των δεδομένων μας
true.head()


# In[7]:


#Ενώνουμε τους πίνακες fake και true
frames = [true, fake]
result = pd.concat(frames)

#Κρατάμε μόνο τα δεδομένα που θα χρειαστούμε
result.drop('text', axis='columns', inplace=True)
result.drop('subject', axis='columns', inplace=True)
result.drop('date', axis='columns', inplace=True)
result.drop('title', axis='columns', inplace=True)

result.tail(20)


# In[8]:


#Απομονώνουμε τα features
result_feat = pd.DataFrame(result.drop(result.columns[[6,7]], axis=1))
result_feat.head()


# In[9]:


#Ελέγχουμε το target
result['target'].head()


# In[10]:


#Χωρίζουμε τα δεδομένα μας σε δεδομένα εκπαίδευσης/δοκιμής
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(result_feat,result['target'],
                                                    test_size=0.30)


# In[11]:


#Αν δεν υπάρχει ο xgboost, τον εγκαθιστούμε με τις παρακάτω εντολές
#pip install xgboost
#or
#conda install -c conda-forge xgboost

from xgboost import XGBClassifier

#Ο AdaBoostClassifier είναι εναλλακτικός
#adb = AdaBoostClassifier(n_estimators=100, random_state=0)

#Ορίζουμε τον Classifier
xgb = XGBClassifier(n_estimators=100, random_state=0)


# In[12]:


#Εκπαιδεύουμε τον classifier από τα δεδομένα μας
xgb.fit(X_train,y_train)


# In[13]:


#Ζητάμε από τον Classifier να ελέγξει τα δεδομένα δοκιμής αν είναι αληθινά ή ψευδή
pred = xgb.predict(X_test) 
 


# In[14]:


#Ελέγχουμε αν τα πήγε καλά
predictions = [round(value) for value in pred]
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100.0}%")


# In[15]:


#Βάζουμε νέους τίτλους για να ελέγξει ο Classifier αν είναι αληθινοί οι ψεύτικοι
#data = [{'title':'Three things to watch as Joe Biden meets Vladimir Putin'}]
data = [{'title':'Israel strikes in Gaza after fire balloons launched'}]

df = pd.DataFrame(data)

df["to"] = df['title'].str.count("to ")
df["in"] = df['title'].str.count("in ")
df["on"] = df['title'].str.count("on ")
df["On"] = df['title'].str.count("On ")
df["with"] = df['title'].str.count("with ")
df["THE"] = df['title'].str.count("THE ")
df.drop('title', axis='columns', inplace=True)

print(df)


# In[16]:


#Ζητάμε την πρόβλεψη του Classifier
pred = xgb.predict(df)
print(pred)

