# Import Required Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

#Load Dataset

df = pd.read_csv("Superstore.csv", encoding='latin1')
print("\n Initial Date : \n",df.head())
print("\n Data Information \n")
df.info()

# Check for missing values

print("\n Missing values \n",df.isnull().sum())

# Convert Sales and Profit to numeric (if necessary)
df['Sales'] = df['Sales'].replace('[\$,]', '', regex=True).astype(float)
df['Profit'] = df['Profit'].replace('[\$,]', '', regex=True).astype(float)

# Mean sales and profit
mean_values = df.groupby(['Category', 'Region', 'Segment'])[['Sales', 'Profit']].mean()
print("\n Mean Sales and Profit:\n " , mean_values)

# Count of orders
order_counts=df.groupby(['Category', 'Region', 'Segment'])['Order ID'].count()
print("\n Order Counts: \n",order_counts)

# Multiple aggregations
aggregated_sales=df.groupby(['Category', 'Region', 'Segment'])['Sales'].agg(['sum', 'mean', 'max'])
print("\n Aggregated Sales:",aggregated_sales)

# Summary Statistics

print("\n summary Statistics",df.describe())
print("\n Median Values\n",df.median(numeric_only=True))
print("\n Standard Deviation\n",df.std(numeric_only=True))

### Visualization:

# Histograms

numeric_cols = ['Sales', 'Profit', 'Discount', 'Quantity']
df[numeric_cols].hist(bins=30, figsize=(12, 8))
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.suptitle('Histograms of Numeric Features')
plt.show()

# Boxplots

plt.figure(figsize=(12, 6))
for i, col in enumerate(numeric_cols):
    plt.subplot(1, 4, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')
plt.xlabel("Category")
plt.ylabel("Profit")
plt.tight_layout()
plt.show()

# Heatmap and pairplot

sns.pairplot(df[numeric_cols])
plt.suptitle("Pairplot of Numeric Features", y=1.02)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.xlabel("Features")
plt.ylabel("Features")
plt.show()

# Sales & Profit by Sub-Category

category_group = df.groupby('Sub-Category')[['Sales', 'Profit']].sum().sort_values(by='Sales', ascending=False).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(data=category_group, x='Sub-Category', y='Sales')
plt.xticks(rotation=45)
plt.title('Total Sales by Sub-Category')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=category_group, x='Sub-Category', y='Profit')
plt.xticks(rotation=45)
plt.title('Total Profit by Sub-Category')
plt.show()

# Plotly Interactive: Sales vs Profit

fig = px.scatter(df, x='Sales', y='Profit', color='Category', size='Quantity', hover_data=['Sub-Category'])
fig.update_layout(title='Sales vs Profit (Bubble by Quantity)')
fig.show()
 
# Discount vs Profit 

sns.scatterplot(data=df, x='Discount', y='Profit', hue='Category')
plt.title('Profit vs Discount')
plt.show()

