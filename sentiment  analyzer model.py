#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system(' pip install nltk')


# In[2]:


import pandas as pd
import numpy as np

df=pd.read_csv('a1_RestaurantReviews_HistoricDump.csv')
df.head(5)


# In[3]:


df.shape


# # Data  processing

# In[4]:


import re 
import nltk

nltk.download('stopwords')


# In[5]:


from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

all_stopwords= stopwords.words('english')
all_stopwords.remove('not')


# In[6]:


corpus=[]
for i in range (0,900):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    


# In[7]:


corpus


# # Data transformation
# 

# In[8]:


from sklearn.feature_extraction.text import CountVectorizer

cv =CountVectorizer(max_features=1420)


# In[9]:


X= cv.fit_transform(corpus).toarray()
Y=df.iloc[:,-1].values


# In[10]:


import pickle
bow_path='c1_BoW_Sentiment_model.pkl'
pickle.dump(cv,open(bow_path,"wb"))


# In[11]:


#dividing the training and test dataset

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


# # model fitting (NAIVE BAYES)

# In[12]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

classifier.fit(X_train,Y_train)


# In[13]:


import joblib 
joblib.dump(classifier, 'c2_Classifier_Sentiment_Model')


# In[24]:


#model performance (accuracy)
Y_pred=classifier.predict(X_test)

for i in range(len(X_test)):
    if Y_pred[i]==Y_test[i]:
        print("true")
    else:
        print("false")


# In[ ]:




