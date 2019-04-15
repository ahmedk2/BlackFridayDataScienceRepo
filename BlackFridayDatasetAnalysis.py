#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:12:29 2019

@author: Khalid_Northside97
"""


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score

df = pd.read_csv('../Khalid_Northside97/Desktop/BlackFriday.csv')


df.head()

#number of unique users, products and occupations
print("Number of unique users:",len(df.User_ID.unique()))
print("Number of unique products:",len(df.Product_ID.unique()))
print("Number of unique occupations recorded:",len(df.Occupation.unique()))


#check missing values with a heatmap
plt.figure(figsize=(24,16))
sns.heatmap(df.isnull())

#using zero imputation to fill specific fields with 0
df['Product_Category_2'] = df['Product_Category_2'].fillna(0)
df['Product_Category_3'] = df['Product_Category_3'].fillna(0)


#profiles of all 5891 unique users
user_profiles=df[['User_ID','Gender','Age','Occupation','City_Category',
                    'Stay_In_Current_City_Years','Marital_Status']].drop_duplicates()
user_profiles.head()


#popularity of top 10 most purchased items
purchase_by_prod_id=df[['Product_ID','Purchase']].groupby('Product_ID').agg('sum').reset_index().sort_values('Purchase',ascending=False).head(10)
print(purchase_by_prod_id.head(10))

#users who purchased the most and their profiles
purchase_by_user=df[['User_ID','Purchase']].groupby('User_ID').agg('sum').reset_index().sort_values('Purchase',ascending=False).head(10)
temp=df[df['User_ID'].isin(list(purchase_by_user['User_ID']))][['User_ID','Gender','Age','Occupation','City_Category',
                                                          'Stay_In_Current_City_Years','Marital_Status']].drop_duplicates()
temp.merge(purchase_by_user,how='left').sort_values('Purchase',ascending=False)


#heatmap showing correlation relationships between different variables
sns.heatmap(df.corr(),annot=True)


#makes a plot reprsenting purchases of every age group 
sns.countplot(df.Age)

#plots a pie chart showing ratio of purchases made by gender
unique_gender = df.Gender.unique()
countF = df[df['Gender'] == 'F'].count() 
countM = df[df['Gender'] == 'M'].count() 

values= [countF.Gender,countM.Gender]
labels = ['Female', 'Male']
explode = (0.2, 0)
plt.pie(values, labels= values,explode=explode,autopct='%1.1f%%',counterclock=False, shadow=True)
plt.title('Ratio of Purchases Made by Gender')
plt.legend(labels,loc=3)
plt.show()

#total Sales made by each City
sns.countplot(df.City_Category)

#displays purchases for each product cateogry 
#checks which product sold the most in product_category_1
sns.countplot(df.Product_Category_1)
#checks which product sold the most in product_category_2
sns.countplot(df.Product_Category_2)
#checks which product sold the most in product_category_3
sns.countplot(df.Product_Category_3)


#shows purchases made by occupation
sns.countplot(df.Occupation)

#bar graph showing comparison of purchases made by marital status 
#and seperated by gender where 0 represents married and 1 represents 
#single people
sns.countplot(df.Marital_Status)

#People who has less time in the same city go to Black Friday more than
#Those with more time.
#The exception is who has less than 1 (one) year.
sns.countplot(df.Stay_In_Current_City_Years)



#association rules apriori algorithm
#generate a list of products purchased by each User_ID
lst=[]
for item in df['User_ID'].unique():
    lst2=list(set(df[df['User_ID']==item]['Product_ID']))
    if len(lst2)>0:
        lst.append(lst2)

#transforming the transaction data into one-hot encoded data
#if you recieve an error saying "No module named mlxtend.preprocessing"
#write this "!pip install mlxtend" in the IPython console to install mlxtend
#then try to import TransactionEncoder, apriori and association_rules

te=TransactionEncoder()
te_df=te.fit(lst).transform(lst)
df_x=pd.DataFrame(te_df,columns=te.columns_)
print(df_x.head())

#get the frequent items (support >= 0.03)
frequent_items=apriori(df_x,use_colnames=True,min_support=0.03)
frequent_items.head()
frequent_items.describe()

rules=association_rules(frequent_items,metric='lift',min_threshold=1)
rules.antecedents=rules.antecedents.apply(lambda x: next(iter(x)))
rules.consequents=rules.consequents.apply(lambda x: next(iter(x)))
rules=rules.sort_values('lift',ascending=False)



#run the second half of this file after line 138 because of the long wait time
#when using the apriori algorithm 
#Convert Product Category 2 and 3 into integers
df['Product_Category_2']=df['Product_Category_2'].astype(int)
df['Product_Category_3']=df['Product_Category_3'].astype(int)

#remove Product ID and User ID
data=df.drop(['Product_ID','User_ID'],axis=1)

#label categorical variables
data['Gender']=data['Gender'].map( {'M': 0, 'F': 1} ).astype(int)
data['City_Category']=data['City_Category'].map( {'A': 0, 'B': 1, 'C':2} ).astype(int)
data['Age']=data['Age'].map( {'0-17': 0, '18-25': 1, '26-35': 2,'36-45':3,'46-50':4,
                         '51-55':5,'55+':6} ).astype(int)
data['Stay_In_Current_City_Years']=data['Stay_In_Current_City_Years'].map( {'0': 0, '1': 1, '2': 2,'3':3,'4+':4}).astype(int)

#Get an array of feature variables X and target variable y
X=data.drop(['Purchase'],axis=1).values
y=data['Purchase'].values

#Select features to keep based on percentile of the highest scores
Selector_f = SelectPercentile(f_regression, percentile=25)
Selector_f.fit(X,y)

#get the scores of all the features
name_score=list(zip(data.drop(['Purchase'],axis=1).columns.tolist(),Selector_f.scores_))
name_score_df=pd.DataFrame(data=name_score,columns=['Feat_names','F_scores'])
name_score_df.sort_values('F_scores',ascending=False)




#making a copy of the data then selecting top 3 features with the highest 
#F-Scores for Linear Regression
data=df.copy()
data=data[['City_Category','Product_Category_1', 
       'Product_Category_3','Purchase']]

#One-Hot Encoding
data=pd.get_dummies(data=data,columns=['City_Category','Product_Category_1','Product_Category_3'])

#Avoid dummy variable trap by removing one category of each categorical feature after encoding but before training
data.drop(['City_Category_A','Product_Category_1_1','Product_Category_3_0'],axis=1,inplace=True)

X=data.drop(['Purchase'],axis=1).values
y=data['Purchase'].values

#spiltting the data into training and testing 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#the linear regreession model is being applied to the training data
regressor=LinearRegression()
regressor.fit(X_train,y_train)

y_pred=regressor.predict(X_test)
print("Prediction\n",y_pred)
print("Actual\n",y_test)

#evalution metrics r-squared, mae and mse are applied to evaluate
#the linear regression model
print("R_squared Score:",regressor.score(X_test,y_test))


mae = mean_absolute_error(y_test,y_pred)
print("MAE:",mae)

print("RMSE:",mean_squared_error(y_test,y_pred)**0.5)




#classification predictor gender random forest classifier
data=df.copy()
data.drop(['User_ID','Product_ID'],axis=1,inplace=True)

data['Gender']=data['Gender'].map( {'M': 0, 'F': 1} ).astype(int)

data['Age']=data['Age'].map( {'0-17': 0, '18-25': 1, '26-35': 2,'36-45':3,'46-50':4,
                         '51-55':5,'55+':6} ).astype(int)

data['City_Category']=data['City_Category'].map( {'A': 0, 'B': 1, 'C':2} ).astype(int)

data['Stay_In_Current_City_Years']=data['Stay_In_Current_City_Years'].map( {'0': 0, '1': 1, '2': 2,'3':3,'4+':4}).astype(int)

X=data.drop(['Gender'],axis=1).values
y=data['Gender'].values


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

#the random forest classifier is applied to the training data

classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
print("Prediction:",y_pred)
print("Actual:",y_test)

#evaluation metrics confusion matrix, precision score and accuracy score
#are used to evaluate the random forest classifier model 
cm=confusion_matrix(y_test,y_pred)
print("Confusion Matrix\n", cm)

print("Precision Score\n",precision_score(y_test,y_pred,average=None))


print("Accuracy Score: ",accuracy_score(y_test,y_pred))


