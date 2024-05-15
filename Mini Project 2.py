#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('pip', 'install seaborn')


# In[13]:


#calling to the dataset
df = pd.read_csv('C:\\Users\\Khalil\\Desktop\\mini project 2\\Resturaunt_Dataset.csv')


# In[6]:


df.head() #looking at the top of the data


# In[7]:


df.describe() #describing the dataset


# In[8]:


df.info() #looking at the types in the dataset


# In[14]:


df.isnull().sum() #checking for nulls in the data set


# In[6]:


#getting the top 4 cuisines according to the dataset listing in decending order
top_cuisines = df['Cuisines'].value_counts().sort_values(ascending=False)
top5_cuisines = top_cuisines.head(5)


# In[208]:


#creating a pie chart to show the percentages according to cuisines
plt.figure(figsize=(8, 6))
plt.pie(top5_cuisines, labels=top5_cuisines.index, autopct='%1.1f%%', startangle=140)
plt.title('Top 4 Cuisines')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
plt.show()


# In[53]:


#narrowing top 3 to top 2 being chinese and indian, because the data set included pizza restaurants as italian restaurants

#now we will be filtering some of the differences between chinese and indian to determine the best for success
Top2 = df[df['Cuisines'].isin(['Indian', 'Chinese'])]
Price_ranges = Top2['Price range']
print(Price_ranges)

#as we can see this didnt give us much information trying a different way


# In[55]:


#trying them one at a time
indian_price = df[df['Cuisines'] == 'Indian']
indian_price_median = indian_price['Price range'].median()
print("Median price range for Indian cuisine:", indian_price_median)

#now for chinese
chinese_price = df[df['Cuisines'] == 'Chinese']
chinese_price_median = chinese_price['Price range'].median()
print("Median price range for Chinese cuisine:", chinese_price_median)


# In[59]:


#just a simple bar graph showing the difference in the median price range for customers
indian_price_median = 3.0  
chinese_price_median = 1.0 

plt.bar(['Indian', 'Chinese'], [indian_price_median, chinese_price_median], color=['orange', 'green'])
plt.xlabel('Cuisines')
plt.ylabel('Median Price Range')
plt.title('Median Price Range for Indian vs Chinese Cuisine')
plt.show()


# In[71]:


import pandas as pd

#importing a second data set to analyze revenue from the resturaunts
file_path = 'C:/Users/Khalil/Desktop/mini project 2/sales_dataset.csv'

df2 = pd.read_csv(file_path)




# In[141]:


# Display the first few rows of the DataFrame
print(df2.head())

# Display the last few rows of the DataFrame
print(df2.tail())

# Get an overview of the DataFrame, including data types and missing values
print(df2.info())

# Check the shape of the DataFrame (number of rows and columns)
print(df2.shape)

# Get summary for numeric columns
print(df2.describe())

# Check for missing values in each column
print(df2.isnull().sum())


# In[153]:


import matplotlib.pyplot as plt

#histogram for retail sales and warehouse sales
plt.hist(df2['YEAR'], bins=20, weights=df2['RETAIL SALES'], alpha=0.5, label='Retail Sales', edgecolor='black')
plt.hist(df2['YEAR'], bins=20, weights=df2['WAREHOUSE SALES'], alpha=0.5, label='Warehouse Sales', edgecolor='black')

plt.xlabel('Year')
plt.ylabel('Sales')
plt.title('Histogram of Retail and Warehouse Sales over Years')
plt.legend()
plt.show()



# In[167]:


# Check for missing values in specific columns
columns_to_check = ['YEAR', 'MONTH', 'RETAIL SALES', 'WAREHOUSE SALES']
missing_values = df2[columns_to_check].isnull().any(axis=1)

# Remove rows with missing values in any of the specified columns
cleaned_df = df[~missing_values]

cleaned_df.describe()



# In[200]:


import pandas as pd

# Assuming you have a DataFrame cleaned_df with columns 'Year', 'Month', 'Retail Sales', and 'Warehouse Sales'
# Sort the DataFrame by 'Year' and 'Month' if necessary

# Calculate sales growth
cleaned_df['Retail Sales Growth'] = (cleaned_df['RETAIL SALES'] - cleaned_df['RETAIL SALES'].shift(1)) / cleaned_df['RETAIL SALES'].shift(1) * 100
cleaned_df['Warehouse Sales Growth'] = (cleaned_df['WAREHOUSE SALES'] - cleaned_df['WAREHOUSE SALES'].shift(1)) / cleaned_df['WAREHOUSE SALES'].shift(1) * 100

# Drop the first row since it will have NaN values due to shift operation
cleaned_df.dropna(inplace=True)

# Now, you have 'Retail Sales Growth' and 'Warehouse Sales Growth' as your target variables


# In[181]:


# was getting an error later in the code, created this section of code to remove and never ending or looping values from rows
cleaned_df = cleaned_df[~np.isinf(cleaned_df['Retail Sales Growth']) & ~np.isinf(cleaned_df['Warehouse Sales Growth'])]
cleaned_df = cleaned_df[(np.abs(cleaned_df['Retail Sales Growth']) < 1e308) & (np.abs(cleaned_df['Warehouse Sales Growth']) < 1e308)]

# dropping any nan values
X = cleaned_df[['Retail Sales Growth', 'Warehouse Sales Growth']]
X.dropna(inplace=True)

# Recalculate 'y' with the updated 'X'
y = ((X.sum(axis=1) / 2) > growth_threshold).astype(int)

# Check data types
print(X.dtypes)

# Display some statistics and values from X_train
print(X_train.describe())
print(X_train.max())
print(X_train.min())




# In[201]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Cleaning
cleaned_df = cleaned_df.dropna()  # Drop rows with missing values
cleaned_df = cleaned_df[~np.isinf(cleaned_df['Retail Sales Growth']) & ~np.isinf(cleaned_df['Warehouse Sales Growth'])]
cleaned_df = cleaned_df[(np.abs(cleaned_df['Retail Sales Growth']) < 1e308) & (np.abs(cleaned_df['Warehouse Sales Growth']) < 1e308)]

# Define features (X) and target variable (y)
X = cleaned_df[['Retail Sales Growth', 'Warehouse Sales Growth']]
y = ((X.sum(axis=1) / 2) > growth_threshold).astype(int)  # Binary variable indicating whether the growth exceeds 10%

# Drop any NaN values introduced during processing
X.dropna(inplace=True)
y = y.loc[X.index]  # Ensure 'y' aligns with 'X'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose and fit the model
model = LogisticRegression()
model.fit(X_train, y_train)  # Fit the model

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy Score:", accuracy)


# In[184]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming cleaned_df is your DataFrame after cleaning and preprocessing steps

# Define features and target as done previously
X = cleaned_df[['Retail Sales Growth', 'Warehouse Sales Growth']]
growth_threshold = 10
y = ((X.sum(axis=1) / 2) > growth_threshold).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)  # Using 100 trees

# Train the model on the training data
rf_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_rf = rf_model.predict(X_test)

# Calculate the accuracy and other metrics
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix = confusion_matrix(y_test, y_pred_rf)
class_report = classification_report(y_test, y_pred_rf)

# Print the results
print("Random Forest Accuracy:", accuracy_rf)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)



# In[202]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Assuming cleaned_df is your DataFrame after cleaning and preprocessing steps

# Define features and target
X = cleaned_df[['Retail Sales Growth', 'Warehouse Sales Growth']]
growth_threshold = 10
y = ((X.sum(axis=1) / 2) > growth_threshold).astype(int)  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
dt_model.fit(X_train, y_train)

# Make predictions on the test data
y_pred_dt = dt_model.predict(X_test)

# Calculate the accuracy and other metrics
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
class_report_dt = classification_report(y_test, y_pred_dt)

# Print the results
print("Decision Tree Accuracy:", accuracy_dt)
print("Confusion Matrix:\n", conf_matrix_dt)
print("Classification Report:\n", class_report_dt)


# In[203]:


import pandas as pd

# Calculate 'Total Sales' by summing 'Warehouse Sales Growth' and 'Retail Sales Growth'
cleaned_df['Total Sales'] = cleaned_df['Warehouse Sales Growth'] + cleaned_df['Retail Sales Growth']

#split the total revenue based on some google information
cleaned_df['Indian Restaurant Revenue'] = cleaned_df['Total Sales'] * 0.61
cleaned_df['Chinese Restaurant Revenue'] = cleaned_df['Total Sales'] * 0.39

# Display the new columns to verify the calculations
print(cleaned_df[['Total Sales', 'Indian Restaurant Revenue', 'Chinese Restaurant Revenue']].head())


# In[205]:


# Calculate 'Total Sales' by summing the entire columns of 'Warehouse Sales Growth' and 'Retail Sales Growth'
total_sales = cleaned_df['Warehouse Sales Growth'].sum() + cleaned_df['Retail Sales Growth'].sum()

# Now calculate 'Indian Restaurant Revenue' and 'Chinese Restaurant Revenue' from the 'Total Sales'
indian_restaurant_revenue = total_sales * 0.61
chinese_restaurant_revenue = total_sales * 0.39

print("Indian Restaurant Revenue:", indian_restaurant_revenue)
print("Chinese Restaurant Revenue:", chinese_restaurant_revenue)


# In[195]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a pairplot of the dataset
sns.pairplot(cleaned_df[['Warehouse Sales Growth', 'Retail Sales Growth', 'Total Sales', 'Indian Restaurant Revenue', 'Chinese Restaurant Revenue']],
             diag_kind='kde',  
             plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'}, 
             corner=True) 

plt.suptitle('Pairplot of Sales and Revenue Variables', size=16)
plt.subplots_adjust(top=0.95)  # Adjust subplot to not overlap with title
plt.show()


# In[196]:


from sklearn.linear_model import LinearRegression
import numpy as np

# checking to see if I have positive coefficients
features = ['Warehouse Sales Growth', 'Retail Sales Growth', 'Indian Restaurant Revenue', 'Chinese Restaurant Revenue']
target = 'Total Sales'

# Preparing the data - ensuring no missing values
cleaned_df = cleaned_df.dropna(subset=features + [target])

# Splitting the features and target
X = cleaned_df[features]
y = cleaned_df[target]

# Creating linear regression
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Get the coefficients
coefficients = model.coef_.ravel()

# Print the coefficients
print("Coefficients of the model:")
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef}")


# In[197]:


import pandas as pd
import matplotlib.pyplot as plt

# shortening variable
indian_revenue = indian_restaurant_revenue
chinese_revenue = chinese_restaurant_revenue

# Growth rate
growth_rate = 0.10  # 10%

# Create a DataFrame to store the projections
years = range(1, 6)  # Next 5 years
projections = pd.DataFrame(index=years, columns=['Indian Restaurant Revenue', 'Chinese Restaurant Revenue'])

# Calculating the projected revenues
for year in years:
    indian_revenue *= (1 + growth_rate)
    chinese_revenue *= (1 + growth_rate)
    projections.at[year, 'Indian Restaurant Revenue'] = indian_revenue
    projections.at[year, 'Chinese Restaurant Revenue'] = chinese_revenue

# Plotting the projections if both restaurants had same rate of growth
plt.figure(figsize=(10, 6))
plt.plot(projections['Indian Restaurant Revenue'], marker='o', linestyle='-', label='Indian Restaurant Revenue')
plt.plot(projections['Chinese Restaurant Revenue'], marker='o', linestyle='-', label='Chinese Restaurant Revenue')

plt.title('Projected Revenue Growth Over the Next 5 Years')
plt.xlabel('Years from Now')
plt.ylabel('Revenue')
plt.xticks(range(1, 6), ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'])
plt.legend()
plt.grid(True)
plt.show()


# In[207]:


import pandas as pd
import matplotlib.pyplot as plt

indian_revenue = indian_restaurant_revenue 
chinese_revenue = chinese_restaurant_revenue                       

#accurate growth rate
indian_growth_rate = 0.10  # 10%
chinese_growth_rate = 0.16  # 16%

# Create a DataFrame to store the projections
years = range(1, 6)  # Next 5 years
projections = pd.DataFrame(index=years, columns=['Indian Restaurant Revenue', 'Chinese Restaurant Revenue'])

# Calculate the projected revenues for each year
for year in years:
    indian_revenue *= (1 + indian_growth_rate)
    chinese_revenue *= (1 + chinese_growth_rate)
    projections.at[year, 'Indian Restaurant Revenue'] = indian_revenue
    projections.at[year, 'Chinese Restaurant Revenue'] = chinese_revenue

# Plotting the projections
plt.figure(figsize=(10, 6))
plt.plot(projections['Indian Restaurant Revenue'], marker='o', linestyle='-', color='red', label='Indian Restaurant Revenue')
plt.plot(projections['Chinese Restaurant Revenue'], marker='o', linestyle='-', color='blue', label='Chinese Restaurant Revenue')

plt.title('Projected Revenue Growth Over the Next 5 Years')
plt.xlabel('Years from Now')
plt.ylabel('Revenue')
plt.xticks(range(1, 6), ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'])
plt.legend()
plt.grid(True)
plt.show()


# In[199]:


import pandas as pd
import matplotlib.pyplot as plt

# Starting values based on your last known revenue figures
indian_revenue = indian_restaurant_revenue  # Replace with actual starting value
chinese_revenue = chinese_restaurant_revenue  # Replace with actual starting value

# Growth rates
indian_growth_rate = 0.10  # 10%
chinese_growth_rate = 0.16  # 16%

# changed from above to 10 years
years = range(1, 11)  
projections = pd.DataFrame(index=years, columns=['Indian Restaurant Revenue', 'Chinese Restaurant Revenue'])

# Calculate the projected revenues for each year
for year in years:
    indian_revenue *= (1 + indian_growth_rate)
    chinese_revenue *= (1 + chinese_growth_rate)
    projections.at[year, 'Indian Restaurant Revenue'] = indian_revenue
    projections.at[year, 'Chinese Restaurant Revenue'] = chinese_revenue

# Plotting the projections
plt.figure(figsize=(12, 8))
plt.plot(projections['Indian Restaurant Revenue'], marker='o', linestyle='-', color='red', label='Indian Restaurant Revenue')
plt.plot(projections['Chinese Restaurant Revenue'], marker='o', linestyle='-', color='blue', label='Chinese Restaurant Revenue')

plt.title('Projected Revenue Growth Over the Next 10 Years')
plt.xlabel('Years from Now')
plt.ylabel('Revenue')
plt.xticks(range(1, 11), [f'Year {i}' for i in range(1, 11)])  # Dynamically generate year labels
plt.legend()
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[72]:


df2.describe()


# In[73]:


df2.info()


# In[76]:


#only using the sales columns from the dataset and printing missing values 
selected_columns = ['RETAIL SALES', 'WAREHOUSE SALES']
selected_df = df2[selected_columns]
missing_values = selected_df.isnull().sum()
print("Missing values in each selected column:")
print(missing_values)


# In[77]:


#optionally I wanted to check the percentage of missing values
total_rows = len(selected_df)
missing_percentage = (missing_values / total_rows) * 100
print("\nPercentage of missing values in each selected column:")
print(missing_percentage)


# In[86]:


#had to use .loc because i was getting a copy warning .loc changes the original df and not a copy of it
selected_df.loc[:, 'total_sales'] = selected_df['RETAIL SALES'] + selected_df['WAREHOUSE SALES']

#have the total number as far as yearly revenue for both restaurants
total_sales_sum = selected_df['total_sales'].sum()
print("Total sum of sales:", total_sales_sum)


# In[87]:


total_sales_sum = selected_df['total_sales'].sum()

# indian cuisine percetage of total profits
indian_yearly_revenue = 0.62 * total_sales_sum

# chinese cuisine percentage of total profits
chinese_yearly_revenue = 0.38 * total_sales_sum

print("Indian Restaurant Yearly Revenue:", indian_yearly_revenue)
print("Chinese Restaurant Yearly Revenue:", chinese_yearly_revenue)


# In[127]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 10% increase calculation
indian_increase = 0.10  # 10% increase
indian_yearly_increase = np.full_like(target, indian_increase)

# linear regression model
indian_model = LinearRegression()

# Training the indian cuisine model
indian_model.fit(X[['indian_yearly_revenue']], target)

# predicting revenue for the next 5 years
future_years = np.arange(2023, 2028).reshape(-1, 1)
future_indian_revenue = indian_model.predict(future_years)

#plot
plt.plot(future_years, future_indian_revenue, marker='o')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Predicted Revenue for Indian Restaurants for the Next 5 Years')
plt.grid(True)
plt.show()


# In[126]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 10% increase a year
chinese_increase = 0.10  # 10% increase
chinese_yearly_increase = np.full_like(target, chinese_increase)

# linear regression
chinese_model = LinearRegression()

# training chinese cuisine model
chinese_model.fit(X[['chinese_yearly_revenue']], target)

# revenue prediction
future_years = np.arange(2023, 2028).reshape(-1, 1)
future_chinese_revenue = chinese_model.predict(future_years)

# plot
plt.plot(future_years, future_chinese_revenue, marker='o')
plt.xlabel('Year')
plt.ylabel('Revenue')
plt.title('Predicted Revenue for Chinese Restaurants for the Next 5 Years')
plt.grid(True)
plt.show()


# In[ ]:





# In[ ]:




