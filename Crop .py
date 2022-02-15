#!/usr/bin/env python
# coding: utf-8

# # Crop recommendation with Machine Learning

# ##### Import the required modules

# In[1]:



# Import the required modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # Visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 
sns.set(color_codes=True)
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score


# 
# ###### Import the dataset

# In[2]:


# Import the dataset

df = pd.read_csv("C:/Users/chand/OneDrive/Desktop/Crop_recommendation (2).csv")




# ##### Data Visualisation and exploration

# In[3]:


# To identify the features in the dataset
df.head() 


# In[4]:


df.tail()


# In[5]:


df.describe() # Gives the multiple SummaryStatistics# We can see the mean, min and max values of climatic conditions and NPK nutrients


# In[6]:


df.size


# In[7]:


df.shape


# In[8]:


df.columns 


# ###### Data pre-processing and cleaning

# In[9]:


df['label'].unique() # Different crops in the dataset


# In[10]:


df.dtypes # checking for different data types


# In[11]:


df['label'].value_counts() # Returns a series containing unique values as its index and frequencies as its values


# In[12]:


#checking for null values

df.notnull()


# ##### Visualisation
# 
# 
# #To understand the correlation between different factors we are building a correlation matrix with heatmap
# 
# #Rainfall and nitrogen content have the highest correlation, followed by potassium and phosphorus
# 
# #Correlation refers to how close two variables are to having a linear relationship with each other. Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable. So, when two features have high correlation, we can drop one of the two features.
# 

# In[13]:


sns.heatmap(df.corr(),annot = True)


# In[14]:


# Prepare and Visualise the dataset
   
all_columns = df.columns[:-1]
for column in all_columns:
   plt.figure(figsize=(14,7))
   sns.barplot(x = "label", y = column, data = df)
   plt.xticks(rotation=90)
   plt.title(f"{column} vs Crop Type")
   plt.show()


# ###### Pair plot

# In[15]:


# Bivariate Analysis
sns.set_theme (style = "ticks")
sns.pairplot (df, hue = "label")


# In[16]:


# Pairplot shows the spread of datapoints so helps us to decide which algorithm to use 
 


# ##### Data preparation 

# In[17]:


# Splitting the columns in the dataset into input(features) and Output(target)
#Input data/features

X = df.drop('label', axis=1)

X.head()


# In[18]:


#Output/target
#le=LabelEncoder()
#df['label'] = le.fit_transform(df['label'])
y = df['label']
y.head()


# ##### Train Test Split

# In[19]:


# Train Test Split

X_train,X_test,y_train,y_test = train_test_split (X,y,test_size=0.2, random_state=42)


# ##### Model building

# In[20]:


# Supervised Learning # Data with label


# In[21]:


LogisticRegression
LG = LogisticRegression()


# In[22]:


LG.fit(X_train, y_train)


# In[ ]:





# In[23]:


predictions = LG.predict(X_test)


# In[24]:


print(confusion_matrix(y_test,predictions))


# In[25]:


print(classification_report(y_test,predictions))


# ###### F1 score is the measure of a test’s accuracy. 
# It considers both the prediction p and
# recall r of the test to compute the score.
# 
# Confusion matrix, also known as an error matrix, is a specific table layout that
# allows visualization of the performance of an algorithm, typically a supervised
# learning one. 
# 
# 

# In[26]:


print(accuracy_score(y_test,predictions))


# In[27]:


LG_accuracy = LG.score(X_test,y_test)


# In[28]:


LG_accuracy


# In[ ]:





# ###### Random Forest is a machine learning algorithm that can be used for both Classification and Regression problems
# #It creates a random sampling of decision trees from the training dataset
# #It takes less training time compared to other algorithms
# 

# In[29]:



RF = RandomForestClassifier()
RF.fit(X_train,y_train)


# In[30]:


predictions = RF.predict(X_test)
print(confusion_matrix(y_test,predictions))


# In[31]:


#How can we measure the Error in Classification Problem? :-
#Type I Error (False Positive) and Type II Error (False Negative) help us to identify the accuracy of our Model which can be found with the help of Confusion Matrix. If we sum the value of Type I and Type II Error, we can have a Total Error = False Negative + False Positive.
#Accuracy will be higher if Error is less and vice versa. Better the accuracy, better the performance and that exactly what we want.


# In[32]:


print(classification_report(y_test,predictions))


# In[33]:


print(accuracy_score(y_test,predictions))


# In[34]:


RF_accuracy = RF.score(X_test,y_test)
RF_accuracy


# In[ ]:





# In[35]:


RF.score(X_train,y_train)


# In[36]:


RF.score(X_test,y_test)


# In[37]:


# Here training score is 100 % but the test score is 99.32 % 
# If you find test score being less than the train score then you conclude the model is overfiting the data. 


# In[38]:


# For OPtimization we perform Hyper parameter tuning


# ###### XGBoost Classifier

# In[39]:


#from sklearn.ensemble 
import xgboost as xgb
xgb=xgb.XGBClassifier()


# In[40]:


xgb.fit(X_train,y_train)
pred = xgb.predict(X_test)


# In[41]:


print('acc', accuracy_score(y_test,pred))
print('f1', classification_report(y_test,pred))
print('matrix', confusion_matrix(y_test,pred))


# ##### Predicting the crop using ML

# In[65]:


# Making a prediction

data = np.array([[60, 55, 44 , 23.004459, 82.32, 7.84, 263.96]])
prediction = xgb.predict(data)
print(prediction)


# In[66]:


data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = xgb.predict(data)
print(prediction)


# In[44]:


data = np.array([[117, 32, 34, 26.272418, 52.127394, 6.758793, 127.175293]])
prediction = RF.predict(data)
print(prediction)             


# In[45]:


# Hyperparameter tuning- Settings of an algorithm can be adjusted to optimize performance
# Random search cross validation

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(rf.get_params())


# In[46]:


le=LabelEncoder()
df['label'] = le.fit_transform(df['label'])
y = df['label']
y


# In[47]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[48]:


#To determine if random search yielded a better model, we compare the base model with the best random search model.


# In[ ]:





# In[49]:


#K Nearest Neighbor algorithm falls under the Supervised Learning category and is used for classification (most commonly) and regression. It is a versatile algorithm also used for imputing missing values and resampling datasets. As the name (K Nearest Neighbor) suggests it considers K Nearest Neighbors (Data points) to predict the class or continuous value for the new Datapoint.


# In[50]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import metrics


# In[51]:



X_train, X_test, y_train, y_test= train_test_split(X, y, test_size= 0.2,random_state=32)
sc= StandardScaler()
sc.fit(X_train)
X_train= sc.transform(X_train)
sc.fit(X_test)
X_test= sc.transform(X_test)
X.shape


# In[52]:


knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred= knn.predict(X_test)
metrics.accuracy_score(y_test,y_pred)


# In[ ]:





# ###### Model comparision

# In[53]:


model = []


accuracy = []


# In[54]:


accuracy.append(LG_accuracy)

model.append('Logistic Regression')


# In[55]:


accuracy.append(RF_accuracy)
model.append('Random Forest')


# In[56]:


accuracy.append (xgb)
model.append('xgb')


# In[57]:


model.append (knn)
accuracy.append(metrics.accuracy_score(y_test,y_pred))


# In[58]:


model


# In[59]:


accuracy


# In[60]:


# On comparing the four models, Random Forest algorithm is giving the highest accuracy


# ##### Predicting

# In[67]:


# Making a prediction

data = np.array([[60, 55, 44 , 23.004459, 82.32, 7.84, 263.96]])
prediction = RF.predict(data)
print(prediction)


# In[68]:


# Making a prediction

data = np.array([[60, 55, 44 , 23.004459, 82.32, 7.84, 263.96]])
prediction = xgb.predict(data)
print(prediction)


# In[69]:


data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = xgb.predict(data)
print(prediction)


# In[70]:


data = np.array([[83, 45, 60, 28, 70.3, 7.0, 150.9]])
prediction = RF.predict(data)
print(prediction)


# In[ ]:




