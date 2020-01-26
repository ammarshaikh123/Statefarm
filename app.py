import numpy as np
import pandas as pd

# Import KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

# Import DecisionTreeClassifier from sklearn.tree
from sklearn.tree import DecisionTreeClassifier

# Import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

path = '/Users/apple/Documents/TAMUHACK2020/'
df = pd.read_csv(path + 'US_Accidents_Dec19.csv')

df_tx = df.loc[df['State'] == 'TX']

## preprocess
df_tx['Start_Time'] = pd.to_datetime(df_tx['Start_Time'], errors='coerce')
df_tx['End_Time'] = pd.to_datetime(df_tx['End_Time'], errors='coerce')

# Extract year, month, day, hour and weekday
df_tx['Year']=df_tx['Start_Time'].dt.year
df_tx['Month']=df_tx['Start_Time'].dt.strftime('%b')
df_tx['Day']=df_tx['Start_Time'].dt.day
df_tx['Hour']=df_tx['Start_Time'].dt.hour
df_tx['Weekday']=df_tx['Start_Time'].dt.strftime('%a')

# Extract the amount of time in the unit of minutes for each accident, round to the nearest integer
td='Time_Duration(min)'
df_tx[td]=round((df_tx['End_Time']-df_tx['Start_Time'])/np.timedelta64(1,'m'))

## remove duration longer than 24 hours 
outliers=df_tx[td]>1500
# Set outliers to NAN
df_tx[outliers] = np.nan

# Drop rows with negative td
df_tx.dropna(subset=[td],axis=0,inplace=True)

n=3

median = df_tx[td].median()
std = df_tx[td].std()
outliers = (df_tx[td] - median).abs() > std*n

# Set outliers to NAN
df_tx[outliers] = np.nan

# Fill NAN with median
df_tx[td].fillna(median, inplace=True)

# Print time_duration information
#print('Max time to clear an accident: {} minutes or {} hours or {} days; Min to clear an accident td: {} minutes.'.format(df[td].max(),round(df[td].max()/60), round(df[td].max()/60/24), df[td].min()))

# Set the list of features to include in Machine Learning
feature_lst=['Source','TMC','Severity','Start_Lng','Start_Lat','Distance(mi)','Side','City','County','Timezone','Temperature(F)','Humidity(%)','Pressure(in)', 'Visibility(mi)', 'Wind_Direction','Weather_Condition','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit','Railway','Roundabout','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Hour','Weekday'] # 'Time_Duration(min)'

# Select the dataset to include only the selected features
df_sel=df_tx[feature_lst].copy()

df_sel.dropna(subset=df_sel.columns[df_sel.isnull().mean()!=0], how='any', axis=0, inplace=True)

df_tx_dummy = pd.get_dummies(df_sel,drop_first=True)

# Set the target for the prediction
target='Severity'

# set X and y
y = df_tx_dummy[target]
X = df_tx_dummy.drop(target, axis=1)

# Split the data set into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)




#########
# Random Forest algorithm

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Get the accuracy score
acc=accuracy_score(y_test, y_pred)

# Append to the accuracy list
# accuracy_lst.append(acc)

#
# # Model Accuracy, how often is the classifier correct?
print("[Randon forest algorithm] accuracy_score: {:.3f}.".format(acc))

# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.03
sfm = SelectFromModel(clf, threshold=0.03)

# Train the selector
sfm.fit(X_train, y_train)

feat_labels=X.columns

# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, y_train)

# Apply The Full Featured Classifier To The Test Data
y_pred = clf.predict(X_test)

print(X_test.head(1))

# View The Accuracy Of Our Full Feature Model
print('[Randon forest algorithm -- Full feature] accuracy_score: {:.3f}.'.format(accuracy_score(y_test, y_pred)))

