import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import mysql.connector

#1
# load csv
data = pd.read_csv("heart_disease(in).csv", nrows=900)
data['age'] = pd.to_numeric(data['age'], errors='coerce')
data.loc[data['age'] <= 0, 'age'] = pd.NA

# mysql connection string
connection = mysql.connector.connect(host = 'de300HW', user = 'root', password = 'meow', port = 3306)

cursor = connection.cursor()

cursor.execute("USE mysql;")

cursor.execute("DROP TABLE IF EXISTS heart_disease")

# table schema given csv
schema = """
CREATE TABLE IF NOT EXISTS heart_disease (
    age FLOAT,
    sex INTEGER,
    painloc INTEGER,
    painexer INTEGER,
    relrest INTEGER,
    pncaden INTEGER,
    cp INTEGER,
    trestbps INTEGER,
    htn INTEGER,
    chol INTEGER,
    smoke INTEGER,
    cigs INTEGER,
    years INTEGER,
    fbs INTEGER,
    dm INTEGER,
    famhist INTEGER,
    restecg INTEGER,
    ekgmo INTEGER,
    ekgday INTEGER,
    ekgyr INTEGER,
    dig INTEGER,
    prop INTEGER,
    nitr INTEGER,
    pro INTEGER,
    diuretic INTEGER,
    proto INTEGER,
    thaldur FLOAT,
    thaltime FLOAT,
    met INTEGER,
    thalach INTEGER,
    thalrest INTEGER,
    tpeakbps INTEGER,
    tpeakbpd INTEGER,
    dummy INTEGER,
    trestbpd INTEGER,
    exang INTEGER,
    xhypo INTEGER,
    oldpeak FLOAT,
    slope INTEGER,
    rldv5 INTEGER,
    rldv5e INTEGER,
    ca INTEGER,
    restckm INTEGER,
    exerckm INTEGER,
    restef INTEGER,
    restwm INTEGER,
    exeref INTEGER,
    exerwm INTEGER,
    thal INTEGER,
    thalsev INTEGER,
    thalpul INTEGER,
    earlobe INTEGER,
    cmo INTEGER,
    cday INTEGER,
    cyr INTEGER,
    target INTEGER
);
"""

# execute schema creation query
cursor.execute(schema)
 
# load data into table
for row in data.to_records(index=False):
    row = {column: value if pd.notnull(value) else None for column, value in zip(data.columns, row)}
    placeholders = ', '.join(['%s'] * len(row))
    columns = ', '.join(row.keys())
    sql = f"INSERT INTO heart_disease ({columns}) VALUES ({placeholders})"
    row_values = list(row.values())
    cursor.execute(sql, row_values)

# commit changes
connection.commit()

#2
# retrieve data from db
query = "SELECT * FROM heart_disease;"
cursor.execute(query)

# fetch data from db
data = pd.DataFrame(cursor.fetchall(), columns=data.columns)

#3
# check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

numerical = ['age', 'thaldur', 'thaltime', 'oldpeak', 'tpeakbps', 'tpeakbpd', 'years', 
             'ekgyr', 'ekgmo', 'ekgday', 'cigs', 'restckm', 'exerckm', 'rldv5',
             'rldv5e', 'cmo', 'cday', 'cyr', 'thalsev', 'thalpul', 'thalrest', 'restef', 'restwm',
             'exeref', 'exerwm', 'ca', 'trestbps', 'trestbpd', 'chol', 'thalach', 'oldpeak', 'met']
categorical = ['sex', 'painloc', 'painexer', 'relrest', 'pncaden', 'cp', 'htn', 'smoke',
               'fbs', 'dm', 'famhist', 'restecg', 'dig', 'prop', 'nitr', 'pro', 'diuretic',
               'proto', 'exang', 'xhypo', 'slope', 'dummy', 'earlobe', 'thal']

# imputation
# numerical features: use median, since not as influenced by outliers
for feature in numerical:
    median_value = data[feature].median()
    data[feature] = data[feature].fillna(median_value)
# categorical features: use mode, preserves most frequently present value
for feature in categorical:
    mode_values = data[feature].mode()
    if not mode_values.empty:
        mode_value = mode_values.iloc[0] 
        data[feature] = data[feature].fillna(mode_value)
    else:
        print(f"No mode found for feature '{feature}'.")

#4
# outliers - z-score to detect, replace with median
    # z-score sensitive to mean and sd, so extreme values can affect greatly
    # median mitigates impact of extreme values
z_scores = stats.zscore(data.select_dtypes(include=['int', 'float']))
abs_z_scores = np.abs(z_scores)
outlier_indices = np.where(abs_z_scores > 3)

for i, j in zip(*outlier_indices):
    data.iloc[i, j] = data.iloc[:, j].median()

#5
# statistical measures
statistics = data.describe()
print("Statistical Measures:")
print(statistics)
# analysis
print("Mean age is around 53.48 years, with a standard deviation of around 9.43 years.")
print("Sex mean value of 0.79 suggests that the majority of patients are male, with some variability, as indicated by standard deviation of 0.41")
print("Mean serum cholesterol level is about 242.36 mg/dl, with a standard deviation of 49.33 mg/dl")
print("Mean maximum heart rate achieved is around 149.65 beats per minute, with standard deviation 22.90 bpm.")
print("The mean target value of 0.55 suggests that there is slightly higher prevalence of heart disease in the patients who are listed in the dataset.")

#6
# feature transformations
# standardization for numerical features where mean=0, sd=1
scaler = StandardScaler()
data[numerical] = scaler.fit_transform(data[numerical])
# one-hot encoding for categorical features
data = pd.get_dummies(data, columns=categorical)

#7
# box plots
data.boxplot(column=['age', 'trestbps', 'chol', 'thalach', 'oldpeak'])
plt.show()
print("Generally higher ages and cholesterol levels for patients in the data. The oldpeak data suggests that some patients (higher outliers) experience significant ST segment depression during exercise.")
# scatter plots
plt.scatter(data['age'], data['thalach'], c=data['target'])
plt.xlabel('age')
plt.ylabel('max heart rate')
plt.title('age vs max heart rate')
plt.show()
print("Maximum heart rate achieved seems to decrease as age increases, which is reasonable, since physical fitness tends to decrease with older age.")
plt.scatter(data['trestbps'], data['thalach'], c=data['target'])
plt.xlabel('resting blood pressure (trestbps)')
plt.ylabel('max heart rate (thalach)')
plt.title('resting bp vs max heart rate')
plt.show()
print("There seems to be a slight pattern of high resting blood pressure accompanying low maximum heart rates. This is reasonable, since as people age, they decline in maximum heart rate, but can also have higher resting blood pressure levels, making it harder to reach a higher rate while exercising.")

#8 
# store cleaned data in new table
cursor.execute("DROP TABLE IF EXISTS cleaned_heart_disease")

data = data.dropna()

# create table dynamically based on columns in the CSV file
columns = ", ".join([f"`{col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')}` TEXT" for col in data.columns])
create_table_query = f"CREATE TABLE cleaned_heart_disease ({columns})"
cursor.execute(create_table_query)

# load data into new table
for _, row in data.iterrows():
    placeholders = ", ".join(["%s"] * len(row))
    insert_query = f"INSERT INTO cleaned_heart_disease VALUES ({placeholders})"
    cursor.execute(insert_query, tuple(row))

# commit changes and close connection
connection.commit()
connection.close()