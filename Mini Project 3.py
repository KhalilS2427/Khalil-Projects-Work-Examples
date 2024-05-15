#!/usr/bin/env python
# coding: utf-8

# In[85]:


## Import libraries
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# import keras
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('pip', 'install seaborn')
import seaborn as sns


# In[86]:


#calling to the dataset
df = pd.read_csv('C:\\Users\\Khalil\\Desktop\\mini project 3\\insurance_cost.csv')


# In[87]:


df.head(50) #looking at the top of the data


# In[88]:


df.describe() #describing the dataset


# In[89]:


df.info() #looking at the types in the dataset


# In[90]:


df.isnull().sum() #checking for nulls in the data set


# In[96]:


import seaborn as sns
import matplotlib.pyplot as plt

heatmap_data = df.pivot_table(index='state', values=['full_coverage', 'minimum_coverage', 'difference'], aggfunc='mean')

# Creating Cluster Heatmap
sns.clustermap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f")
plt.title('Insurance Prices Across States')
plt.xlabel('Insurance Types')
plt.ylabel('State')
plt.show()


# In[162]:


# Filtered heatmap data for states Illinois, Indiana, and Ohio
selected_states = ['Illinois', 'Indiana', 'Ohio']
heatmap_data_filtered = heatmap_data.loc[selected_states]

# Cluster heatmap
sns.clustermap(heatmap_data_filtered, cmap='YlGnBu', annot=True, fmt=".2f")
plt.title('Insurance Prices Across Illinois, Indiana, and Ohio')
plt.xlabel('Insurance Types')
plt.ylabel('State')
plt.show()


# In[69]:


import seaborn as sns
import pandas as pd

# Creating pairplot
sns.pairplot(df)

# Show the plot
plt.show(10)


# In[22]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Prepare data for regression
X = df[['full_coverage', 'minimum_coverage']]  # Features
y = df['difference']  # Target variable

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predict differences for all states
df['predicted_difference'] = model.predict(X)

# Create histogram
plt.hist(df['predicted_difference'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Predicted Difference')
plt.ylabel('Frequency')
plt.title('Histogram of Predicted Differences')
plt.grid(True)
plt.show()


# In[24]:


df2 = pd.read_csv('C:\\Users\\Khalil\\Desktop\\mini project 3\\insurance_claims.csv')


# In[82]:


df2.head(10)


# In[100]:


df2.isnull().sum()


# In[101]:


df2.info


# In[114]:


df2.dtypes


# In[115]:


unique_values_counts = df2['incident_state'].value_counts()
print(unique_values_counts)


# In[116]:


df2.head(20)


# In[117]:


from sklearn.model_selection import train_test_split

# Features (X) and target variable (y)
X = df2.drop(columns=['fraud_reported'])
y = df2['fraud_reported']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# You can print the shapes of the training and testing sets to verify the split
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# In[118]:


from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

# Defining pipeline with imputation and logistic regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LogisticRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)



# In[119]:


# Drop features without observed values trying to get better accuracy
X_train_filtered = X_train.drop(columns=['_c39'])
X_test_filtered = X_test.drop(columns=['_c39'])

# Define a pipeline with imputation, scaling, and logistic regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', LogisticRegression(max_iter=1000))  # Increased max_iter
])

# Train the model
pipeline.fit(X_train_filtered, y_train)

# Make predictions
y_pred = pipeline.predict(X_test_filtered)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[120]:


from sklearn.preprocessing import StandardScaler

# Define a pipeline with imputation, scaling, and logistic regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # Add StandardScaler for scaling
    ('model', LogisticRegression(max_iter=1000, solver='sag'))  # Use 'sag' solver
])

# Train the model
pipeline.fit(X_train_filtered, y_train)

# Make predictions
y_pred = pipeline.predict(X_test_filtered)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[121]:


# Create an interaction term between 'age' and 'policy_annual_premium'
df2['age_times_premium'] = df2['age'] * df2['policy_annual_premium']


# In[122]:


from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features for 'age'
poly = PolynomialFeatures(degree=2, include_bias=False)
age_poly = poly.fit_transform(df2[['age']])
df2['age_squared'] = age_poly[:, 1]  # Use the squared term


# In[123]:


from sklearn.preprocessing import StandardScaler

# Define a pipeline with imputation, scaling, and logistic regression
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),  # Add StandardScaler for scaling
    ('model', LogisticRegression(max_iter=1000, solver='sag'))  # Use 'sag' solver
])

# Train the model
pipeline.fit(X_train_filtered, y_train)

# Make predictions
y_pred = pipeline.predict(X_test_filtered)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[126]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Separate features (X) and target variable (y)
X = df2.drop(columns=['fraud_reported'])
y = df2['fraud_reported']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate model
print("Decision Tree Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))


# In[164]:


import matplotlib.pyplot as plt
import numpy as np

# Get predicted probabilities of fraud for the test set
y_pred_prob_dt = dt_classifier.predict_proba(X_test)[:, 1]

# Sort the test samples based on the incident_date
sorted_indices = np.argsort(X_test['incident_date'])
sorted_dates = X_test['incident_date'].iloc[sorted_indices]
sorted_probabilities = y_pred_prob_dt[sorted_indices]

# Plot the predicted probabilities over time
plt.figure(figsize=(10, 6))
plt.plot(sorted_dates, sorted_probabilities, marker='o')
plt.xlabel('Incident Date')
plt.ylabel('Predicted Probability of Fraud')
plt.title('Predicted Probability of Fraud Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()


# In[136]:


# checking counts in column
total_count = df2['policy_number'].count()

print("Total count:", total_count)



# In[155]:


# First, filter the DataFrame to include only rows where 'fraud_reported' is not 0
filtered_df = df2[df2['fraud_reported'] != 0]

# Next, get the unique policy numbers from the filtered DataFrame
policy_numbers_with_fraud = filtered_df['policy_number'].unique()

# Now, you can exclude these policy numbers from the original DataFrame
excluded_df = df2[~df2['policy_number'].isin(policy_numbers_with_fraud)]

# Print the shape of the excluded DataFrame to verify the number of rows
print("Shape of Policies Flagged For Fraud:", policy_numbers_with_fraud.shape)


# In[156]:


# Convert the list of policy numbers to a pandas Series
policy_numbers_series = pd.Series(policy_numbers_with_fraud)

# Filter the original DataFrame to include only rows where 'policy_number' is in the list of policy numbers with fraud reported
rows_with_fraud = df2[df2['policy_number'].isin(policy_numbers_series)]

# Print the first few rows of the filtered DataFrame
rows_with_fraud.head(247)


# In[160]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a countplot
plt.figure(figsize=(8, 6))
ax = sns.countplot(data=rows_with_fraud, x='policy_state', palette='Set2')
plt.title('Number of Frauds Reported by Policy State')
plt.xlabel('Policy State')
plt.ylabel('Number of Frauds Reported')

# Add text annotations for the legend
plt.text(0, rows_with_fraud['policy_state'].value_counts().max() + 10, 'Indiana', color='green', ha='center')
plt.text(1, rows_with_fraud['policy_state'].value_counts().max() + 10, 'Ohio', color='orange', ha='center')
plt.text(2, rows_with_fraud['policy_state'].value_counts().max() + 10, 'Illinois', color='purple', ha='center')

plt.show()


# In[161]:


import numpy as np
import matplotlib.pyplot as plt

growth_rates = {0: 0.08, 1: 0.12, 2: 0.06}

# Map state codes to state names
state_names = {0: 'Indiana', 1: 'Ohio', 2: 'Illinois'}

# Initialize plot
plt.figure(figsize=(10, 6))

# Iterate over each state
for state_code, growth_rate in growth_rates.items():
    # Calculate projected number of fraud reports over the next 5 years
    projected_frauds = [rows_with_fraud['policy_state'].value_counts().loc[state_code]]
    for year in range(1, 6):
        projected_frauds.append(projected_frauds[-1] * (1 + growth_rate))

    # Plot the projected number of fraud reports
    plt.plot(np.arange(0, 6), projected_frauds, label=state_names[state_code])

# Add labels and title
plt.title('Projected Increase in Fraud Reports Over the Next 5 Years')
plt.xlabel('Years')
plt.ylabel('Number of Fraud Reports')
plt.xticks(np.arange(0, 6))
plt.legend()

# Show plot
plt.grid(True)
plt.show()


# In[ ]:




