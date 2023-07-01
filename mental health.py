#!/usr/bin/env python
# coding: utf-8

# In[93]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


data = pd.read_csv(r"C:\Users\Admin\Data analytics project\Untitled Folder\survey.csv")


# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.info()


# In[7]:


data['Country'].value_counts().plot(kind='bar',figsize=(10,8))


# In[8]:


data.drop(['Country','state','Timestamp','comments'], axis = 1, inplace=True)


# In[9]:


data.isnull().sum()


# In[10]:


data['self_employed'].value_counts()


# In[11]:


data['self_employed'].fillna('No', inplace=True)


# In[12]:


data['work_interfere'].value_counts()


# In[13]:


data['work_interfere'].fillna('N/A',inplace=True)


# In[14]:


data['Age'].value_counts().plot(kind='bar',figsize=(10,8))


# In[15]:


data.drop(data[(data[ 'Age' ]>60) | (data[ 'Age' ]<18)].index, inplace=True)


# In[16]:


data['Gender'].value_counts().plot(kind='bar',figsize=(10,8))


# In[17]:


data['Gender'].replace(['Male ', 'male', 'M', 'm', 'Male', 'Cis Male','Man', 'cis male','Mail','Male-ish', 'Male (CIS)','Cis Man', 'msle','Malr','Mal','maile', 'Make',], 'Male', inplace = True)
data[ 'Gender']. replace([ 'Female ','female','F','femail','Woman', 'Female','Cis Female','cis-female/femme','Femake', 'Female (cis)','woman', ], 'Female', inplace=True)


# In[18]:


data['Gender'].replace(['Female (trans)', 'queer/she/they', 'non-binary', 'fluid', 'queer', 'Agender', 'Androgyne', 'Trans-female', 'male learning androgynous', 'A little about you', 'Nah', 'All', 'ostensibly male', 'unsure what that really means', 'Genderqueer', 'Enby', 'p', 'Neuter', 'something kinda male?', 'Guyish', 'Trans woman'], 'Non-Binary', inplace=True)


# In[19]:


sb.distplot(data["Age"])
plt.title("Distribution - Age")
plt.xlabel("Age")


# In[20]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 1)
sb.countplot(x='self_employed', hue='treatment', data=data)
plt.title('Employment Type')

plt.show()


# In[21]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 2)
sb.countplot(x='family_history', hue='treatment', data=data)
plt.title('Family_History')

plt.show()


# In[22]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 3)
sb.countplot(x='work_interfere', hue='treatment', data=data)
plt.title('Work Interfere')

plt.show()


# In[23]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 4)
sb.countplot(x='remote_work', hue='treatment', data=data)
plt.title('Work Type')

plt.show()


# In[24]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 5)
sb.countplot(x='tech_company', hue='treatment', data=data)
plt.title('Company')

plt.show()


# In[25]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 6)
sb.countplot(x='benefits', hue='treatment', data=data)
plt.title('Benefits')

plt.show()


# In[26]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 7)
sb.countplot(x='care_options', hue='treatment', data=data)
plt.title('Care Options')

plt.show()


# In[27]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 8)
sb.countplot(x='mental_vs_physical', hue='treatment', data=data)
plt.title('Equal importance to mental and physical health')

plt.show()


# In[28]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 9)
sb.countplot(x='wellness_program', hue='treatment', data=data)
plt.title('Wellness Program')

plt.show()


# In[29]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 10)
sb.countplot(x='anonymity', hue='treatment', data=data)
plt.title('Anonymity')

plt.show()


# In[30]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 11)
sb.countplot(x='leave', hue='treatment', data=data)
plt.title('Leave')

plt.show()


# In[31]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 12)
sb.countplot(x='mental_health_consequence', hue='treatment', data=data)
plt.title('Mental Health Consequence')

plt.show()


# In[32]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 13)
sb.countplot(x='phys_health_consequence', hue='treatment', data=data)
plt.title('Physical health Consequence')

plt.show()


# In[33]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 14)
sb.countplot(x='coworkers', hue='treatment', data=data)
plt.title('Discussion with coworkers')

plt.show()


# In[34]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 15)
sb.countplot(x='supervisor', hue='treatment', data=data)
plt.title('Discussion with supervisor')

plt.show()


# In[35]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 16)
sb.countplot(x='mental_health_interview', hue='treatment', data=data)
plt.title('Discussion with Interviewer')

plt.show()


# In[36]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 17)
sb.countplot(x='phys_health_interview', hue='treatment', data=data)
plt.title('Discussion with Interviewer')

plt.show()


# In[37]:


plt.figure(figsize=(10, 40))
plt.subplot(9, 2, 18)
sb.countplot(x='obs_consequence', hue='treatment', data=data)
plt.title('Consequence After Disclosure')

plt.show()


# In[38]:


data.describe(include='all')


# In[39]:


X = data.drop('treatment', axis = 1)
y = data['treatment']


# In[40]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder


# In[41]:


X = data. drop ('treatment', axis = 1)
y = data ['treatment']


# In[42]:


ct = ColumnTransformer ([('oe',OrdinalEncoder(),['Gender', 'self_employed', 'family_history', 'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                                                 'benefits', 'care_options', 'wellness_program', 'seek_help', 'anonymity', 'leave', 'mental_health_consequence',
                                                 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
                                                 'mental_vs_physical', 'obs_consequence' ])], remainder='passthrough')


# In[43]:


X = ct.fit_transform(X)


# In[44]:


le = LabelEncoder ()
y = le.fit_transform(y)


# In[91]:


import joblib
joblib.dump(ct,'feature_values')


# In[92]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=49)


# In[47]:


X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[48]:


from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier 
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report,auc


# In[49]:


model_dict = {}

model_dict['Logistic regression']= LogisticRegression (solver='liblinear', random_state=49)
model_dict['KNN Classifier' ] = KNeighborsClassifier ()
model_dict[ 'Decision Tree Classifier' ] = DecisionTreeClassifier (random_state=49)
model_dict ['Random Forest Classifier'] = RandomForestClassifier (random_state=49)
model_dict ['AdaBoost Classifier' ] = AdaBoostClassifier (random_state=49)
model_dict ['Gradient Boosting Classifier' ] = GradientBoostingClassifier (random_state=49)
model_dict ['XGB Classifier'] = XGBClassifier (random_state=49)


# In[50]:


def model_test (X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train,y_train)
    y_pred = model.predict (X_test)
    accuracy = accuracy_score (y_test,y_pred)
    print('====================================={}======================================='.format(model_name))
    print('Score is : {}'.format (accuracy))
    print()


# In[52]:


for model_name, model in model_dict.items ():
    model_test(X_train, X_test, y_train, y_test, model,model_name)


# In[53]:


abc = AdaBoostClassifier (random_state=99)
abc.fit (X_train,y_train)
pred_abc = abc.predict (X_test)
print ('Accuracy of AdaBoost=', accuracy_score (y_test,pred_abc))


# In[65]:


from sklearn.model_selection import RandomizedSearchCV
params_abc = {'n_estimators': [int(x) for x in np.linspace(start = 1, stop = 50, num = 15)],
              'learning_rate': [(0.97 + x / 100) for x in range(0, 8)],
             }
abc_random = RandomizedSearchCV (random_state=49, estimator=abc, param_distributions = params_abc,n_iter =50,cv=5,n_jobs=-1)


# In[66]:


params_abc


# In[67]:


abc_random.fit(X_train, y_train)


# In[70]:


abc_random.best_params_


# In[71]:


abc_tuned = AdaBoostClassifier (random_state=49, n_estimators=11, learning_rate=1.02)
abc_tuned.fit (X_train,y_train)
pred_abc_tuned = abc_tuned.predict (X_test)
print ('Accuracy of Adaboost (tuned)=' , accuracy_score (y_test,pred_abc_tuned))


# In[72]:


cf_matrix = confusion_matrix(y_test, pred_abc)
sb.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt=' .2%') 
plt.title( 'Confusion Matrix of AdaBoost Classifier') 
plt.xlabel('Predicted') 
plt.ylabel('Actual')


# In[78]:


from sklearn import metrics
fpr_abc, tpr_abc, thresholds_abc = roc_curve (y_test, pred_abc)
roc_auc_abc = metrics.auc(fpr_abc, tpr_abc)
plt.plot (fpr_abc, tpr_abc, color='orange', label= 'ROC curve (area = %8.2f)' % roc_auc_abc)
plt.plot ([0, 1], [0, 1], color='blue', linestyle='--') 
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.0]) 
plt.title('ROC Curve')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel ('True Positive Rate (Sensitivity)' )
plt.legend(loc="lower right")
plt.show()
roc_curve(y_test,pred_abc)


# In[83]:


fpr_abc_tuned, tpr_abc_tuned, thresholds_abc_tuned = roc_curve (y_test, pred_abc_tuned)
roc_auc_abc_tuned = metrics.auc (fpr_abc_tuned, tpr_abc_tuned)
plt.plot (fpr_abc_tuned, tpr_abc_tuned, color='orange',label= 'ROC curve (area = %0.2f) ' % roc_auc_abc_tuned)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0]) 
plt.ylim([0.0, 1.0]) 
plt.title('ROC Curve' )
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity' )
plt.legend(loc="lower right")
plt.show()
roc_curve(y_test,pred_abc_tuned)


# In[84]:


print(classification_report(y_test,pred_abc))


# In[85]:


print(classification_report(y_test,pred_abc_tuned))


# In[88]:


import pickle
filename = 'model.pkl'
pickle.dump('abc_tuned',open('model.pkl','wb'))


# In[ ]:





# In[ ]:




