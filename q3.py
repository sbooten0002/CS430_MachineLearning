"""""""""""""""""""""
CS430 Quiz
Savanna Booten
11-1-2022
"""""""""""""""""""""




""" Import Libraries """
######################################################
# Helpful 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Model builder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# Model Perfomance
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

######################################################




""" Create/Import Dataset """
######################################################
# Make up a random dataset

""" If you have to make your own dataset
num_rows = 15
num_cols = 4
col_names = ['a','b','c','d']

data_i_made = np.random.randint(0,100,size=(num_rows, num_cols))
df = pd.DataFrame(data=data_i_made, columns=col_names)

class_types = ['class1','class1','class1','class1','class1','class1','class1','class2','class2','class2','class2','class2','class2','class2','class2']
df["class_type"] = class_types
display(df)
"""

# If given a csv file, do this instead:
df = pd.read_csv('my_csv_file.csv')

# Add the different class names here
# i.e. types of fruit
class_names = ["class1", "class2"]


# If you have to add column names because the csv didn't come with it:
#my_column_names_as_a_list = ['column1','column2','column3','column4','column5']
#df = pd.read_csv('my_csv_file.csv',names=my_column_names_as_a_list)
######################################################




""" Preprocessing - Impute Missing Values """
######################################################
print("There are", df.isna().sum().sum(), "missing values in this dataset")
# Initialize SimpleImputer object 
impute_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

# Automatically find all columns that have nans/NAs
list_of_columns_with_nans = df.columns[df.isna().any()].tolist()

# Calculate mean(s) of column(s)
impute_mean = impute_mean.fit(df[list_of_columns_with_nans])

# Replace missing values with the calculated means
df[list_of_columns_with_nans] = impute_mean.transform(df[list_of_columns_with_nans]).round()
######################################################




""" Preprocessing - Categorical Variable Encoding """
######################################################
# Initialize encoder object
encoder = preprocessing.OneHotEncoder()

# Transform categorical variable into dummy variable
# need to put y var. in place of class_type
df = encoder.fit_transform(df[['class_type']])

# Convert back to dataframe
df1 = pd.DataFrame(df.toarray(), columns=encoder.get_feature_names_out(), dtype=int)

display(df1)
######################################################




""" Preprocessing - Create Training/Testing Sets """
######################################################
# Split data randomly into two sets at an 80:20 ratio 

train_set = train_test_split(df, test_size=0.2, random_state=42)[0]
test_set = train_test_split(df, test_size=0.2, random_state=42)[1]

# Tell program where the X and Y columns are
column_num_with_Y = 4
column_num_with_X_START = 0
column_num_with_X_END = 4

Y_train = train_set.iloc[:,column_num_with_Y] 
X_train = train_set.iloc[:,column_num_with_X_START:column_num_with_X_END]

Y_test = test_set.iloc[:,column_num_with_Y] 
X_test = test_set.iloc[:,column_num_with_X_START:column_num_with_X_END]
######################################################




""" Preprocessing - Min-Max Normalize Data """
######################################################
# Initialize the MinMaxScaler object
min_max_scaler = preprocessing.MinMaxScaler()

# Transform numeric values in the training and testing sets
# Do this for X only since Y is categorical
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
display(X_train)
######################################################


# Classifier1  
# For each classifier algorithm, copy this block
# Rename classifier1, y_predict1, y_train_predict1 to classifier2 instead of 1 and so on
# Then change the algorthim in 'classifier#':
# Naive Bayes: GaussianNB()
# Random Forest: RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
# K-Nearest Neighbors: 
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" Model Building - Fit Model to Data"""
######################################################
# Initialize Classifier object
classifier1 = GaussianNB()

# Fit the GaussianNB model to training data 
# also apply the predict method using testing data
y_predict1 = classifier1.fit(X_train, Y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (Y_test != y_predict1).sum()))
######################################################




""" Model Building - Performance"""
######################################################
y_train_predict1 = cross_val_predict(classifier1, X_train, Y_train, cv=3)

# Generate a Classification report
print(classification_report(Y_train, y_train_predict1, target_names=class_names))
######################################################
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




""" Plotting """
######################################################
classifier1_accuracy = accuracy_score(Y_train, y_train_predict1)
classifier2_accuracy  = accuracy_score(Y_train, y_train_predict2)

classifier1_precision = precision_score(Y_train, y_train_predict1,average="binary", pos_label="R")
classifier2_precision = precision_score(Y_train, y_train_predict2,average="binary", pos_label="R")

classifier1_recall = recall_score(Y_train, y_train_predict1,average="binary", pos_label="R")
classifier2_recall = recall_score(Y_train, y_train_predict2,average="binary", pos_label="R")

# width of the bars
barWidth = 0.3
 
# Choose the height of the blue bars
bars1 = [classifier1_accuracy, classifier1_precision, classifier1_recall]
 
# Choose the height of the cyan bars
bars2 = [classifier2_accuracy, classifier2_precision, classifier2_recall]
 
# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'dodgerblue', edgecolor = 'black', capsize=7, label='classifier1')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'deeppink', edgecolor = 'black', capsize=7, label='classifier2')
 
# general layout
plt.xticks([r + barWidth/2 for r in range(len(bars1))], ['Accuracy', 'Precision', 'Recall'])
plt.ylabel('Percent')
plt.legend()
 
# Show graphic
plt.show()
######################################################
