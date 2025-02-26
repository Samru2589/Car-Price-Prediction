# Car Price Prediction
# 1) Import Libraries
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.ion()  # Enable interactive mode
import seaborn as sns

# 2) Load the dataset

df = pd.read_csv(r"C:\Users\Samruddhi Yadav\Downloads\CarPrice_Assignment.csv")


#### Understand the dataset

print(df.head())        # First few rows
print(df.info())        # Data types and missing values
print(df.describe())    # Statistical summary
print(df.columns)       # List of all columns

# 3) Data Cleaning

print(df.isnull().sum())  # Check for missing values
df.dropna(inplace=True)   # Remove rows with missing values (if any)


# 4) Exploratory Data Analysis

plt.figure(figsize=(8,5))
sns.histplot(df['price'], bins=30, kde=True)
plt.title('Car Price Distribution')


#### Scateer plot

plt.figure(figsize=(8,5))
sns.scatterplot(x=df['horsepower'], y=df['price'])
plt.title("Horsepower vs Price")


df['CarName']

# Create the new column companyname and remove the space, only it contain the first name.
df['companyname']=df['CarName'].apply(lambda x:x.split(' ')[0])
df=df.drop(['CarName'], axis=1)
df.head()

df.companyname=df.companyname.str.lower()
df.companyname.unique()

# As we see above the companyname column have the incorrect car names so we replace it with the correct once.
def company_name_rep(df,a,b):
    return df.companyname.replace(a,b)

company_name_rep(df,'maxda','mazda')
company_name_rep(df,'porcshce','porsche')
company_name_rep(df,'toyouta','toyota')
company_name_rep(df,'vokswagen','volkswagen')
company_name_rep(df,'vw','volkswagen')
company_name_rep(df,'alfa-romero','alfa-romeo')

print(df.companyname.unique())
print("\nNumber of unique car companies: ",df.companyname.nunique())

# Plotting the histogram plot and box plot for the target variable column.

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(14,8),dpi=100)
plt.subplot(121)
plt.title('Car Price: Distribution Plot',fontweight='bold',fontsize=18)
sns.histplot(df['price'], kde=False,color='b',bins=10)
plt.xlabel('Price of Car', fontstyle='italic', fontsize=14)
plt.subplot(122)
plt.title('Car Price: Box Plot',fontweight='bold',fontsize=18)
sns.boxplot(y=df['price'],color='g')
plt.xlabel('Price of Car', fontstyle='italic', fontsize=14)

# Seperate columns as "Categorical" and "Numerical" Columns.
numeric_columns= df.select_dtypes(include =['int64','float64'])
numeric_columns.head()

categorical_columns= df.select_dtypes(include =['object'])
categorical_columns.head()

Car_Price= pd.get_dummies(df,drop_first=True)
Car_Price.head()

# Create a model of price prediction

from sklearn.model_selection import train_test_split
X=Car_Price.drop('price',axis=1)
y=Car_Price['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# Apply the correct method 'fit_transform'
X_train_scaled_data = scaler.fit_transform(X_train)
X_test_scaled_data = scaler.fit_transform(X_test)

from sklearn.linear_model import LinearRegression
model= LinearRegression()
model.fit(X_train_scaled_data, y_train)

from sklearn.feature_selection import RFE
import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score

rfe=RFE(estimator= model,n_features_to_select=10)
rfe= rfe.fit(X_train, y_train)

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

col_rfe_sup= X_train.columns[rfe.support_]
col_rfe_sup

X_train_rfe = X_train[col_rfe_sup]

# Defining a function for creating models

def build_model(y_train,X_train_rfe):
   
    
    X = sm.add_constant(X_train_rfe) #Adding the constant
    lm = sm.OLS(y_train,X_train_rfe).fit() # Fitting the model
    print(lm.summary()) # Model summary
    return X_train_rfe

import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def mod_vif(X_train_set):
    car_vif = pd.DataFrame()
    
    # Ensure that X_train_set is a Pandas DataFrame, even if it was converted to a NumPy array
    if isinstance(X_train_set, pd.DataFrame) == False:
        X_train_set = pd.DataFrame(X_train_set)
    
    # Create a DataFrame for storing VIF values
    car_vif['Features'] = X_train_set.columns  # Ensure we get the column names
    car_vif['VIF'] = [variance_inflation_factor(X_train_set.values, i) for i in range(X_train_set.shape[1])]
    
    # Round VIF values for better readability
    car_vif['VIF'] = round(car_vif['VIF'], 2)
    
    return car_vif

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
##Building Model 1: 
X_train_rfe = build_model(y_train, X_train_rfe)

car_vif = mod_vif(X_train_rfe)
car_vif
##Checking VIFs of independent variables
#mod_vif(X_train_rfe)

y_pred= model.predict(X_test_scaled_data)

print(y_pred)

# Deploy the model of Car Price Prediction

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Convert categorical columns to dummy variables
df = pd.get_dummies(df, drop_first=True)  # Drop first to avoid dummy variable trap



# Splitting the dataset
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Streamlit UI
st.title("Car Price Prediction 🚗💰")

st.sidebar.header("Enter Car Details")

# Get user input
features = []
for col in X.columns:
    if df[col].dtype == 'object':
        features.append(st.sidebar.selectbox(f"Select {col}:", df[col].unique()))
    else:
        features.append(st.sidebar.number_input(f"Enter {col}:", min_value=float(df[col].min()), max_value=float(df[col].max())))

# Predict button
if st.sidebar.button("Predict Price"):
    # Convert input into DataFrame
    user_input = pd.DataFrame([features], columns=X.columns)

    # Scale user input
    user_input_scaled = scaler.transform(user_input)

    # Load model
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    # Make prediction
    prediction = model.predict(user_input_scaled)[0]

    # Display prediction
    st.success(f"Estimated Car Price: ${prediction:,.2f}")

