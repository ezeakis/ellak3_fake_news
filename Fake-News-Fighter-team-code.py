#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Φέρνουμε τη βιβλιοθήκη numpy και την ονομάζουμε np
import numpy as np

#Φέρνουμε τη βιβλιοθήκη pandas και την ονομάζουμε pd
import pandas as pd


# In[22]:


#Φορτώνουμε τα δεδομένα του αρχείου True.csv στη μεταβλητή "true"
fake = pd.read_csv("Fake.csv")

fake["to"] = fake['title'].str.count("to ")
fake["in"] = fake['title'].str.count("in ")
fake["on"] = fake['title'].str.count("on ")
fake["On"] = fake['title'].str.count("On ")
fake["with"] = fake['title'].str.count("with ")
fake["THE"] = fake['title'].str.count("THE ")

fake['target'] = 0

fake['Validity'] = "fake"

#Εμφανίζουμε τις πέντε πρώτες γραμμές των δεδομένων μας
fake.head()


# In[23]:


#Φορτώνουμε τα δεδομένα του αρχείου True.csv στη μεταβλητή "true"
true = pd.read_csv("True.csv")

true["to"] = true['title'].str.count("to ")
true["in"] = true['title'].str.count("in ")
true["on"] = true['title'].str.count("on ")
true["On"] = true['title'].str.count("On ")
true["with"] = true['title'].str.count("with ")
true["THE"] = true['title'].str.count("THE ")


true['target'] = 1

true['Validity'] = "true"

#Εμφανίζουμε τις πέντε πρώτες γραμμές των δεδομένων μας
true.head()


# In[24]:


frames = [true, fake]

result = pd.concat(frames)

result.drop('text', axis='columns', inplace=True)
result.drop('subject', axis='columns', inplace=True)
result.drop('date', axis='columns', inplace=True)
result.drop('title', axis='columns', inplace=True)

result.tail(20)


# In[25]:


result_feat = pd.DataFrame(result.drop(result.columns[[6,7]], axis=1))


# In[26]:


result_feat.head()


# In[27]:


result['target'].head()


# In[28]:


from sklearn.model_selection import train_test_split


# In[29]:


X_train, X_test, y_train, y_test = train_test_split(result_feat,result['target'],
                                                    test_size=0.30)


# In[30]:


#pip install xgboost
#or
#conda install -c conda-forge xgboost

from xgboost import XGBClassifier


# In[31]:


#adb = AdaBoostClassifier(n_estimators=100, random_state=0)

xgb = XGBClassifier(n_estimators=100, random_state=0)


# In[32]:


xgb.fit(X_train,y_train)


# In[33]:


pred = xgb.predict(X_test) 
 


# In[34]:


"""evaluate predictions"""

predictions = [round(value) for value in pred]

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, predictions)

#print(accuracy)


print(f"Accuracy: {accuracy * 100.0}%")


# In[42]:


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


# In[43]:


pred = xgb.predict(df)

print(pred)


# In[ ]:




