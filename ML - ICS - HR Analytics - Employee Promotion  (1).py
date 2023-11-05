#!/usr/bin/env python
# coding: utf-8

# # HR Analystics - Employee Promotion

# HR analytics is a process of collecting and analyzing data about people at work. It aims to answer critical questions and make data-driven decisions that improve the organization's workforce and business performance.
# HR analytics can also be called people analytics, workforce analytics, or talent analytics.
# HR analytics can help with managing employee behavior, performance, productivity, engagement, development, and interactions. It can also help with hiring, firing, and promoting employees.
# Promotions are announced after the evaluation and this leads to delay in transition to new roles.
# 
# The company needs help in identifying the eligible candidates at a particular checkpoint so that they can expedite the entire promotion cycle.
# 

# # Importing the Libraries

# In[1]:


get_ipython().system('pip install imblearn')
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')


# # Importing HR Analytics: Employee Promotion Dataset

# In[2]:


dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')


# In[3]:


#View first five rows of the dataset

dataset_train.head()                                         


# In[4]:


#View last five rows of the dataset

dataset_train.tail()


# In[5]:


#View the shape of the training dataset

dataset_train.shape


# In[6]:


#View the shape of the testing dataset

dataset_test.shape


# # Checking the datatype of the dataset

# In[7]:


#Checking the data types in Training dataset

dataset_train.info()


# Observation -
# 
# 1. There are 54808 rows and 13 columns
# 2. There are null values in the dataset
# 3. awards_won? column is not named correctly

# In[8]:


#Checking the data types in Testing dataset

dataset_test.info()


# Observation - 
# 1. There are 23490 rows and 12 columns
# 2. There are null values in the dataset
# 3. awards_won? column is not named correctly

# In[9]:


#Summary Statistics of the dataset

dataset_train.describe()


# Observations
# 
# 1. Age of the employee range is 20yrs - 60 yrs and the average age of the employees is 34yrs
# 2. The length of service of an employee ranges from 1 year to 37 years
# 3. The average training score ranges from 39 to 99 and the average is 63
# 4. The percentage of promoted employees is 8.5%

# In[10]:


dataset_train.describe(include=['object']).T


# Observations
# 
# 1. Sales & Marketing department has the highest number of employees
# 2. Most employees are from region_2
# 3. Most of the employees have Bachelor's degree
# 4. 70% of the employees are men
# 5. 56% of the employees are recruited from other rectruitment channels

# In[11]:


#Plotting correlation

plt.figure(figsize = (12, 10))
sns.heatmap(dataset_train.corr(), annot=True, square=False, linewidth=0.8)


# Observations
# 
# 1. Some of the variables are correlated like age and length of service (0.66)
# 2. There is correlation between awards won and being promoted which is 0.2

# # Expolatory Data Analysis (EDA)

# In[12]:


train = dataset_train.copy()
test = dataset_test.copy()


# In[13]:


#Column names

train.columns


# Target Variable

# In[14]:


#Plotting target Variable

sns.set_style('darkgrid')
sns.catplot(x ='is_promoted', kind='count', data=train)
plt.show()


# Observation
# 
# - The number of promoted employees is less than the unprommoted employees

# # Data Distribution

# #Data distribution
# 
# train.hist(bins = 20, figsize = (20,10), color = 'g')

# Observations
# 
# - Age
# 1. The mean is greater than the median, i.e mean is 35 and median 33
# 2. Age of the employees ranges from 20 years to 60 years
# 3. Most of the employees ages range from 30 yrs and 40 yrs
# 
# - Length of Service
# 1. Mean(5.8) > median(5.0)
# 2. Length of service ranges from 1 yr to 37 yrs
# 3. The length of service of most employees is centered between 1 and 6 yrs
# 
# - Average training score
# 1. Mean(63.38) > median(60.00)
# 2. Average training score ranges from 39 and 99

# # Department

# In[15]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='department', kind='count', data=train, palette='husl')
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()


# Observations
# 
# 1. Top three departments with the most number of employees are:
# - Sales & Marketing
# - Operations
# - Procurement
# 
# 2. R&D department has the lowest number of employees in the organisation

# In[16]:


plt.figure(figsize=(12, 10))
sns.catplot(x='department', hue='is_promoted', kind='count', data=train, palette='husl')
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()


# In[17]:


train.groupby('department')['is_promoted'].sum()


# Education

# In[18]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='education', kind='count', data=train, palette='Set1')
plt.show()


# Observations
# 
# - More than 35000 employees hold a bachelor's degree
# - At least 15000 employees have a Master's and Phd

# In[19]:


plt.figure(figsize=(12, 10))
sns.catplot(x='education', hue='is_promoted', kind='count', data=train, palette='Set1')
plt.show()


# In[20]:


train.groupby('education')['is_promoted'].sum()


# # Gender

# In[21]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='gender', kind='count', data=train, palette='Set2')
plt.show()


# Observation
# 
# - Male employees account for more than 35000 employees in the company
# - The number of female employees is slightly above 15000

# In[22]:


plt.figure(figsize=(12, 10))
sns.catplot(x='gender', hue='is_promoted', kind='count', data=train, palette='Set2')
plt.show()


# In[23]:


train.groupby('gender')['is_promoted'].sum()


# # Recruitment Channel

# In[24]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='recruitment_channel', kind='count', data=train, palette='Paired')
plt.show()


# Observation
# 
# - Most of the employees are recruited using other recruitment

# In[25]:


plt.figure(figsize=(12, 10))
sns.catplot(x='recruitment_channel', hue='is_promoted', kind='count', data=train, palette='Paired')
plt.show()


# In[26]:


train.groupby('recruitment_channel')['is_promoted'].sum()


# # No of trainings
# 
# 

# In[27]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='no_of_trainings', kind='count', data=train, palette='hls')
plt.show()


# Observation
# 
# - Most of the employees have at least attended 1 training.

# In[28]:


plt.figure(figsize=(12, 10))
sns.catplot(x='no_of_trainings', hue='is_promoted', kind='count', data=train,palette='hls')
plt.show()


# In[29]:


train.groupby('no_of_trainings')['is_promoted'].sum()


# # Previous Year Rating

# In[30]:


plt.figure(figsize=(12,8))
sns.catplot(x ='previous_year_rating', kind='count', data=train, palette='Set3')
plt.show()


# Observation
# 
# - Most of the employees have been in the company for 3 years

# In[31]:


plt.figure(figsize=(12, 10))
sns.catplot(x='previous_year_rating', hue='is_promoted', kind='count', data=train, palette='Set3')
#plt.xticks(rotation=45, horizontalalignment='right')
plt.show()


# In[32]:


train.groupby('previous_year_rating')['is_promoted'].sum()


# # Awards won

# In[33]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='awards_won?',hue='is_promoted', kind='count', data=train, palette='plasma')
plt.show()


# In[34]:


train.groupby('awards_won?')['is_promoted'].sum()


# # Region

# In[35]:


plt.figure(figsize= (40 , 14))
sns.catplot(x ='region', kind='count', data=train, palette='husl')
plt.xticks(rotation=45, horizontalalignment='right')
plt.show()


# In[36]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='education', hue='department', kind='count', data=train, palette='Paired')
plt.show()


# Observation
# 
# - Employees from sales & marketing department have a bachelor's degree

# In[37]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='age', hue='department', kind='count', data=train)
plt.show()


# In[38]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='age', hue='is_promoted', kind='count', data=train, palette='icefire')
plt.show()


# In[39]:


plt.figure(figsize=(12, 10))
sns.catplot(x ='recruitment_channel', hue='department', kind='count', data=train, palette='Paired')
plt.show()


# Most sales & marketing department employees are recruited through other recruitment channels
# 
# 

# In[40]:


sns.pairplot(train)
plt.show()


# # Data Cleaning

# - Check for null values
# - Rename columns
# - Encode columns

# Renaming columns

# In[41]:


train.rename(columns = {'awards_won?':'awards_won'}, inplace = True)

train.columns


# In[42]:


test.rename(columns = {'awards_won?':'awards_won'}, inplace = True)

test.columns


# # Null values

# In[43]:


train.isnull().sum()


# In[44]:


(train.isnull().mean())*100


# In[45]:


test.isnull().sum()


# In[46]:


#Percentage of null values in test dataset

(test.isnull().mean())*100


# # Handle null values

# In[47]:


from sklearn.impute import SimpleImputer


# In[48]:


impute = SimpleImputer(strategy='median')
train['previous_year_rating']=impute.fit_transform(train[['previous_year_rating']])
test['previous_year_rating']=impute.fit_transform(test[['previous_year_rating']])


# In[49]:


impute1 = SimpleImputer(strategy='most_frequent')
train['education']=impute1.fit_transform(train[['education']])
test['education']=impute1.fit_transform(test[['education']])


# # Dropping column
# 

# In[50]:


dataset_train = train.drop('employee_id',1)
dataset_test = test.drop('employee_id',1)


# # One Hot Encoding

# In[51]:


#Encoding columns, train data
train_cat = dataset_train.select_dtypes(include=['object']).columns
train = pd.get_dummies(dataset_train, drop_first=True, columns=train_cat)
train.head()


# In[52]:


# Encoding columns, test data
test_cat = dataset_test.select_dtypes(include=['object']).columns
test = pd.get_dummies(dataset_test, drop_first=True, columns=test_cat)
test.head()


# # Modelling

# In[53]:


#Import ML Libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import f1_score,recall_score
from sklearn.metrics import plot_confusion_matrix
from imblearn.over_sampling import SMOTE 
from sklearn.model_selection import train_test_split


# In[54]:


#Defining independent and dependent variables
X = train.drop('is_promoted', axis=1)
y = train.is_promoted
X.shape, y.shape


# SMOTE() The dataset is imbalanced, so we use SMOTE() to balance the dataset

# In[55]:


#Resampling

X_res, y_res  = SMOTE().fit_resample(X, y.values.ravel())


# In[56]:


#Train test split

X_train, X_valid, y_train, y_valid = train_test_split(X_res, y_res, test_size = 0.3, random_state = 42)


# In[57]:


#StandardScaler follows Standard Normal Distribution (SND). 
#It makes mean = 0 and scales the data to unit variance.

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_valid = sc.transform(X_valid)
test = sc.transform(test)


# # Logistic Regression Model

# In[58]:


#Logistic Regression model
log_model = LogisticRegression() #Define
log_model.fit(X_train, y_train) #fit
predict = log_model.predict(X_valid) #predict


# In[94]:


print("Train data Accuracy :", log_model.score(X_train, y_train))
print("Test data Accuracy :", log_model.score(X_valid, y_valid))


# In[59]:


print(classification_report(y_valid, predict))


# Observations
# 
# - Precision:- Out of all the employees that the model predicted would get promoted, only 92% actually did.
# - Recall:- Out of all the employees that actually did get promoted, the model only predicted this outcome correctly for 83% of those employees.
# - F1 Score: The model predicted an 87% chance of employees being promoted
# - Support:- Among the employees in the test dataset, 15186 did not get promoted and 14898 did get promoted.
# 
# 
# GridSearchCv()
# 
# - This is a tuning technique that attempts to compute the optimum values od hyperparameters

# # Decision Tree Model

# In[60]:


max_depth_range = np.arange(1, 40)
tree_param = [{'criterion': ['entropy', 'gini'], 'max_depth': max_depth_range}]
clf_tree = GridSearchCV(DecisionTreeClassifier(), tree_param, cv=5, scoring='f1_weighted')
clf_tree.fit(X_train, y_train)


# In[61]:


clf_tree.best_params_


# In[62]:


tree_model=DecisionTreeClassifier(criterion='entropy', max_depth=39)
tree_model.fit(X_train, y_train)


# In[63]:


tree_pred = tree_model.predict(X_valid)


# In[64]:


plot_confusion_matrix(tree_model,X_valid, y_valid,normalize='true')


# In[65]:


print("Train data Accuracy :", tree_model.score(X_train, y_train))
print("Test data Accuracy :", tree_model.score(X_valid, y_valid))


# In[66]:


print(classification_report(y_valid, tree_pred))


# Observations
# 
# - Precision:- Out of all the employees that the model predicted would get promoted, only 89% actually did.
# - Recall:- Out of all the employees that actually did get promoted, the model only predicted this outcome correctly for 95% of those employees.
# - F1 Score: The model predicted a 92% chance of employees being promoted
# - Support:- Among the employees in the test dataset, 15186 did not get promoted and 14898 did get promoted.

# # RandomForest Model

# In[93]:


rf_model = RandomForestClassifier(random_state=1)
max_depth_range = np.arange(1, 40)
rf_param = [{'criterion': ['entropy', 'gini'], 'max_depth': max_depth_range}]
clf_rf = GridSearchCV(rf_model, rf_param, cv= 5,scoring='f1_weighted')
clf_rf.fit(X_train, y_train) 


# In[81]:


clf_rf.best_params_


# In[82]:


rf_model=RandomForestClassifier(criterion='gini',max_depth=39)
rf_model.fit(X_train, y_train)


# In[83]:


rf_pred=rf_model.predict(X_valid)


# In[84]:


plot_confusion_matrix(rf_model,X_valid, y_valid,normalize='true')


# In[85]:


print("Train data Accuracy :", rf_model.score(X_train, y_train))
print("Test data Accuracy :", rf_model.score(X_valid, y_valid))


# In[86]:


print(classification_report(y_valid, rf_pred))


# Observations
# 
# - Precision:- Out of all the employees that the model predicted would get promoted, only 96% actually did.
# - Recall:- Out of all the employees that actually did get promoted, the model only predicted this outcome correctly for 94% of those employees.
# - F1 Score: The model predicted an 95% chance of employees being promoted
# - Support:- Among the employees in the test dataset, 15186 did not get promoted and 14898 did get promoted.

# # KNN

# In[87]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[88]:


classifier = KNeighborsClassifier()
classifier.fit(X_train, y_train)


# In[89]:


pred = classifier.predict(X_test)


# # Importance of features

# In[90]:


from sklearn.feature_selection import mutual_info_classif

importances = mutual_info_classif(X_res, y_res)
feat_importances = pd.Series(importances, train.columns[0:len(train.columns)-1])
feat_importances


# In[91]:


plt.figure(figsize=(12, 10))
feat_importances.plot(kind='barh', color='teal')
plt.show()


# Observation
# 
# - Top 3 factors that drive promotion:
# 2. Previous year rating
# 3. Average training score
# 4. Number of trainings

# # Based on the obtained results, it is inferred that Logistic Regression offered the highest accuracy being 90% 
