#!/usr/bin/env python
# coding: utf-8

# The goal of this project is to apply various Machine Learning algorithms on the Pima Indians Diabetes Dataset to detect diabetes in a patient.</font> <br>
# In here we applied - 1) Logistic Regression 2) Support Vector Machine 3) K- Nearest Neighbor 4) Random Forest  5) Naive Bayes Classifier and 6) Gradient Boost Classifier. <br>
# We compare the accuracy of the models and conclude which model is best suited for detecting diabetes over the dataset.

# #### <font color = black>  Step 1: Import the necessary libraries </font> 

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### <font color = black> Step 2: Load the dataset </font>

# In[2]:


df = pd.read_csv('/Users/sriharshithaayyalasomayajula/Desktop/Projects/Diabetes_Prediction/Dataset/diabetes.csv')
df.head()


# In[3]:


#describe the data
df.describe()


# In[4]:


#information of the dataset
df.info()


# In[5]:


#checking for any null values  
df.isnull().values.any()


# #### <font color = black> Step 3: Visualize the data </font>

# In[6]:


#histogram
df.hist(bins = 10,figsize = (10,10), color = '#dc8b48')
plt.show()


# In[7]:


#correlation
sns.heatmap(df.corr())


# In the above plot we can see that the skin thickness, insulin, pregnencies and age are completely independent of eachother. <br>
# We can also observe that age and pregencies have negative correlation. <br>
# We consider the total outcome in each target with 0 and 1
# where,
# - 0 means no diabetes
# - 1 means diabetes

# In[8]:


sns.countplot(y = df['Outcome'],palette = 'Set2')


# In[9]:


sns.set(style = "ticks")
sns.pairplot(df, hue = "Outcome")


# Box plot for outlier visualization

# In[10]:


sns.set(style = "whitegrid")
df.boxplot(figsize = (15,6), color = '#678052')


# In[11]:


#box plot
sns.set(style = "whitegrid")
sns.set(rc = {'figure.figsize':(4,2)})

sns.boxplot(x = df['Insulin'],color = '#d75952')
plt.show()

sns.boxplot(x = df['BloodPressure'], color = '#d75952')
plt.show()

sns.boxplot(x = df['DiabetesPedigreeFunction'], color = '#d75952')
plt.show()


# #### <font color = black> Step 4: Remove the outliers </font>

# In[12]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3-Q1

print("---Q1--- \n",Q1)
print("\n---Q3--- \n",Q3)
print("\n---IQR---\n",IQR)

print((df < (Q1 - 1.5 * IQR))|(df > (Q3 + 1.5 * IQR)))


# In[13]:


#Remove the outliers
df_out = df[~((df < (Q1 - 1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
df.shape,df_out.shape


# We observe that more than 80 records are deleted.

# #### <font color = black> Step 5: Visualize the data after removing the outliers </font>

# In[14]:


sns.set(style = "ticks")
sns.pairplot(df_out, hue = "Outcome")
plt.show()


# #### <font color = black> Step 6: Feature Extraction </font>

# In[15]:


X = df_out.drop(columns = ['Outcome'])
y = df_out['Outcome']


# #### <font color = black> Step 7: Split the data into train and test (80:20) </font>

# In[16]:


from sklearn.model_selection import train_test_split
train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=0.2)

print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)


# In[17]:


from sklearn.metrics import confusion_matrix,accuracy_score,make_scorer
from sklearn.model_selection import cross_validate

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

#cross validation purpose
scoring = {'accuracy': make_scorer(accuracy_score),'prec': 'precision'}
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn)}

def display_result(result):
    print("TP: ",result['test_tp'])
    print("TN: ",result['test_tn'])
    print("FN: ",result['test_fn'])
    print("FP: ",result['test_fp'])


# #### <font color = black> Step 8: Build and Train the models </font>

# #### <font color = purple> 1) Logistic Regression </font>

# In[18]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

acc=[]
roc=[]

clf = LogisticRegression(max_iter = 500)
clf.fit(train_X,train_y)
y_pred = clf.predict(test_X)

#find accuracy
ac = accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc = roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result = cross_validate(clf,train_X,train_y,scoring = scoring,cv = 10)
display_result(result)
pd.DataFrame(data = {'Actual':test_y,'Predicted':y_pred}).head()


# #### <font color = purple> 2) Support Vector Machine </font>

# In[19]:


from sklearn.svm import SVC

clf = SVC(kernel='linear')
clf.fit(train_X,train_y)
y_pred = clf.predict(test_X)

#find accuracy
ac = accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc = roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result = cross_validate(clf,train_X,train_y,scoring = scoring,cv = 10)
display_result(result)
pd.DataFrame(data = {'Actual':test_y,'Predicted':y_pred}).head()


# #### <font color = purple> 3) K- Nearest Neighbor </font>

# In[20]:


from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(train_X,train_y)
y_pred = clf.predict(test_X)

#find accuracy
ac = accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc = roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result = cross_validate(clf,train_X,train_y,scoring = scoring,cv = 10)
display_result(result)
pd.DataFrame(data = {'Actual':test_y,'Predicted':y_pred}).head()


# #### <font color = purple> 4) Random Forest  </font>

# In[21]:


from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
clf.fit(train_X,train_y)
y_pred = clf.predict(test_X)

#find accuracy
ac = accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc = roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result = cross_validate(clf,train_X,train_y,scoring = scoring,cv = 10)
display_result(result)
pd.DataFrame(data = {'Actual':test_y,'Predicted':y_pred}).head()


# #### <font color = purple> 5) Naive Bayes Classifier </font>

# In[22]:


from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()
clf.fit(train_X,train_y)
y_pred = clf.predict(test_X)

#find accuracy
ac = accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc = roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result = cross_validate(clf,train_X,train_y,scoring = scoring,cv = 10)
display_result(result)
pd.DataFrame(data = {'Actual':test_y,'Predicted':y_pred}).head()


# #### <font color = purple> 6) Gradient Boost Classifier <font>

# In[23]:


from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=50,learning_rate=0.2)
clf.fit(train_X,train_y)
y_pred = clf.predict(test_X)

#find accuracy
ac = accuracy_score(test_y,y_pred)
acc.append(ac)

#find the ROC_AOC curve
rc = roc_auc_score(test_y,y_pred)
roc.append(rc)
print("\nAccuracy {0} ROC {1}".format(ac,rc))

#cross val score
result = cross_validate(clf,train_X,train_y,scoring = scoring,cv = 10)
display_result(result)
pd.DataFrame(data = {'Actual':test_y,'Predicted':y_pred}).head()


# #### <font color = black> Step 9: Visualizing the accuracy and ROC of the algorithms using a bar graph </font> 

# In[24]:


ax = plt.figure(figsize = (9,4))
plt.bar(['Logistic Regression','SVM','KNN','Random Forest','Naivye Bayes','Gradient Boosting'],acc,label='Accuracy', color = ['#678052','#d75952','#8282e1','#914c4d','#dc8b48','#000200'])
plt.ylabel('Accuracy Score')
plt.xlabel('Algortihms')
plt.show()

ax = plt.figure(figsize = (9,4))
plt.bar(['Logistic Regression','SVM','KNN','Random Forest','Naivye Bayes','Gradient Boosting'],roc,label='ROC AUC', color = ['#678052','#d75952','#8282e1','#914c4d','#dc8b48','#000200'])
plt.ylabel('ROC AUC')
plt.xlabel('Algortihms')
plt.show()


# Conclusion: By applying various algorithms we observed the following:
# - Random Forest has highest accuracy of 80% and ROC_AUC curve 74%. <br>
# - A model can be improved by fine-tuning it.<br>
# - In our model around 30% are diabetic and 40% are not diabetic.
# 
