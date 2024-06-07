from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.svm import SVC
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, create_map, monotonically_increasing_id
from itertools import chain
import boto3
import os
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, LinearSVC
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from sklearn.pipeline import Pipeline
from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler

# Default args for DAG
default_args = {
    'owner': 'rachel',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 6, 6),
}

dag = DAG(
    'rachel_hw4',
    default_args=default_args,
    description='A simple DAG to process heart disease data',
    schedule_interval=timedelta(days=1),
)

local_path = "/tmp/heart_disease(in).csv"
clean_sklearn_path = "/tmp/heart_disease(in)_clean_sklearn.csv"
clean_spark_path = "/tmp/heart_disease(in)_clean_spark.csv"
fe_sklearn_path = "/tmp/heart_disease(in)_fesklearn.csv"
fe_spark_path = "/tmp/heart_disease(in)_fespark.csv"
scrape_path = "/tmp/heart_disease(in)_scrape.csv"
merged_path = "/tmp/heart_disease(in)_merged.csv"

def load_data_s3(**kwargs):
    s3 = boto3.client('s3')
    bucket_name = 'de300-airflow-rachel'
    file_key = 'heart_disease(in).csv' 
    s3.download_file(bucket_name, file_key, local_path)
    return local_path

def create_spark_session():
    spark = SparkSession.builder \
        .appName("Heart Disease Prediction") \
        .getOrCreate()
    return spark

# clean and impute data - sklearn
def sklearn_clean_and_impute_data(**kwargs):
    data = pd.read_csv(local_path)

    # 1
    retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 
                         'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 
                         'exang', 'oldpeak', 'slope', 'target']
    data = data[retain].copy()

    # 2
    # a. painloc and painexer: replace missing vals with most frequent val
    data['painloc'] = data['painloc'].fillna(data['painloc'].mode()[0])
    data['painexer'] = data['painexer'].fillna(data['painexer'].mode()[0])

    # b. trestbps: vals < 100 mm Hg with median
    data.loc[data['trestbps'] < 100, 'trestbps'] = data['trestbps'].median()
    data['trestbps'] = data['trestbps'].fillna(data['trestbps'].median())

    # c. oldpeak: vals < 0 and > 4 with median
    data.loc[(data['oldpeak'] < 0) | (data['oldpeak'] > 4), 'oldpeak'] = data['oldpeak'].median()
    data['oldpeak'] = data['oldpeak'].fillna(data['oldpeak'].median())

    # d. thaldur and thalach: replace missing vals with median
    data['thaldur'] = data['thaldur'].fillna(data['thaldur'].median())
    data['thalach'] = data['thalach'].fillna(data['thalach'].median())

    # e. fbs, prop, nitr, pro, diuretic: replace missing vals and vals > 1 with 0
    clean = ['fbs', 'prop', 'nitr', 'pro', 'diuretic']
    data[clean] = data[clean].fillna(0)
    data[clean] = data[clean].map(lambda x: 0 if x > 1 else x)

    # f. exang and slope: replace missing vals with mode
    data['exang'] = data['exang'].fillna(data['exang'].mode()[0])
    data['slope'] = data['slope'].fillna(data['slope'].mode()[0])

    data['sex'] = data['sex'].fillna(data['sex'].mode()[0])
    data['cp'] = data['cp'].fillna(data['cp'].mode()[0])  

    data.to_csv(clean_sklearn_path, index=False)
    return clean_sklearn_path

# clean and impute data - spark
def spark_clean_and_impute_data(**kwargs):
    spark = create_spark_session()
    data = spark.read.csv(local_path, header=True, inferSchema=True)

    # Select relevant columns and clean data
    retain = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'smoke', 
            'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 
            'exang', 'oldpeak', 'slope', 'target']
    data = data.select(retain)

    # Fill missing values and clean data
    data = data.fillna({
        'painloc': data.groupBy().agg({"painloc": "max"}).collect()[0][0],
        'painexer': data.groupBy().agg({"painexer": "max"}).collect()[0][0],
        'trestbps': data.approxQuantile("trestbps", [0.5], 0)[0],
        'oldpeak': data.approxQuantile("oldpeak", [0.5], 0)[0],
        'thaldur': data.approxQuantile("thaldur", [0.5], 0)[0],
        'thalach': data.approxQuantile("thalach", [0.5], 0)[0],
        'sex': data.groupBy().agg({"sex": "max"}).collect()[0][0],
        'cp': data.groupBy().agg({"cp": "max"}).collect()[0][0],
    })
    data = data.withColumn('trestbps', when(col('trestbps') < 100, lit(120)).otherwise(col('trestbps')))
    data = data.withColumn('oldpeak', when((col('oldpeak') < 0) | (col('oldpeak') > 4), lit(1)).otherwise(col('oldpeak')))
    data = data.withColumn('fbs', when(col('fbs') > 1, lit(0)).otherwise(col('fbs')))
    data = data.withColumn('prop', when(col('prop') > 1, lit(0)).otherwise(col('prop')))
    data = data.withColumn('nitr', when(col('nitr') > 1, lit(0)).otherwise(col('nitr')))
    data = data.withColumn('pro', when(col('pro') > 1, lit(0)).otherwise(col('pro')))
    data = data.withColumn('diuretic', when(col('diuretic') > 1, lit(0)).otherwise(col('diuretic')))

    data.write.csv(clean_spark_path, header=True, mode='overwrite')
    return clean_spark_path

def scrape(**kwargs):
    # scrape smoking rates by age from source 1
    def scrape_smoking_rates_by_age(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        tables = soup.find_all('table')
        smoking_data = {}
        for table in tables[1]:
            rows = table.find_all('tr')
            for row in rows[1:]:
                ths = row.find_all('th')
                tds = row.find_all('td')
                age_range = ths[0].text.strip()
                smoking_rate = float(tds[9].text.strip())
                if 'and over' in age_range:
                    min_age = int(age_range.split()[0])
                    max_age = 120  # assuming 120 as an upper limit for age
                else:
                    min_age, max_age = map(int, age_range.split('–'))
                for age in range(min_age, max_age + 1):
                    smoking_data[age] = smoking_rate
        return smoking_data

    # scrape smoking rates by age and sex from source 2
    def scrape_smoking_rates_by_age_and_sex(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # sex
        cards = soup.find_all("div", class_="card-body")
        gender_data = {}
        for card in cards[2:3]:
            rows = card.find_all('li', class_='main')
            for row in rows:
                text = row.text.strip()
                gender = text.split()[6]
                rate = float(text.split()[7].strip('()%'))
                gender_data[gender] = rate
        # age
        age_data = {}
        for card in cards[3:4]:
            rows = card.find_all('li')
            for row in rows:
                text = row.text.strip()
                age_range = text.split()[7]
                if 'and older' in text:
                    min_age = int(age_range.split()[0])
                    max_age = 120  # assuming 120 as an upper limit for age
                else:
                    min_age, max_age = map(int, age_range.split('–'))
                    rate = float(text.split()[9].strip('()%'))
                for age in range(min_age, max_age + 1):
                    age_data[age] = rate
        return gender_data, age_data

    # source 1
    source1 = 'https://www.abs.gov.au/statistics/health/health-conditions-and-risks/smoking/latest-release'
    smoking_source1 = scrape_smoking_rates_by_age(source1)

    # source 2
    source2 = 'https://www.cdc.gov/tobacco/data_statistics/fact_sheets/adult_data/cig_smoking/index.htm'
    gender_data, age_data = scrape_smoking_rates_by_age_and_sex(source2)

    spark = create_spark_session()
    data = spark.read.csv(clean_spark_path, header=True, inferSchema=True)
    broadcast_smoking_source1 = spark.sparkContext.broadcast(smoking_source1)
    broadcast_gender_data = spark.sparkContext.broadcast(gender_data)
    broadcast_age_data = spark.sparkContext.broadcast(age_data)

    # Create mappings for smoke imputation
    smoking_map_source1 = create_map([lit(x) for x in chain(*broadcast_smoking_source1.value.items())])
    smoking_map_age = create_map([lit(x) for x in chain(*broadcast_age_data.value.items())])
    men_rate = broadcast_gender_data.value['men']
    women_rate = broadcast_gender_data.value['women']

    # Impute smoke_source1
    data = data.withColumn('smoke_source1', 
                        when(col('smoke').isNull(), smoking_map_source1[col('age')])
                        .otherwise(col('smoke')))

    # Impute smoke_source2
    data = data.withColumn('smoke_source2', 
                        when(col('smoke').isNull() & (col('sex') == 0), smoking_map_age[col('age')])
                        .when(col('smoke').isNull() & (col('sex') == 1), smoking_map_age[col('age')] * (men_rate / women_rate))
                        .otherwise(col('smoke')))

    # Drop original smoke column
    data = data.drop('smoke')

    data = data.withColumn('id', monotonically_increasing_id())

    data.write.csv(scrape_path, header=True, mode='overwrite')
    return scrape_path

def fe_sklearn(**kwargs):
    data = pd.read_csv(clean_sklearn_path)

    data['thalach_squared'] = data['thalach'] ** 2

    data['id'] = data.index

    data.to_csv(fe_sklearn_path, index=False)
    
    return fe_sklearn_path

def fe_spark(**kwargs):
    spark = create_spark_session()
    data = spark.read.csv(clean_spark_path, header=True, inferSchema=True)

    data = data.withColumn('thaldur_squared', col('thaldur') ** 2)

    data = data.withColumn('id', monotonically_increasing_id())
    
    data.write.csv(fe_spark_path, header=True, mode='overwrite')

    return fe_spark_path

def merge(**kwargs):
    spark = create_spark_session()
    df1 = pd.read_csv(fe_sklearn_path)
    df2 = spark.read.csv(fe_spark_path, header=True, inferSchema=True).toPandas()
    df3 = spark.read.csv(scrape_path, header=True, inferSchema=True).toPandas()

    merged_df = pd.merge(df1, df2, on='id', how='inner')
    new_merged_df = pd.merge(merged_df, df3, on='id', how='inner')

    new_merged_df.to_csv(merged_path, index=False)

    return merged_path

# Function to train and evaluate models using sklearn
def sklearn_lr(**kwargs):
    data = pd.read_csv(fe_sklearn_path)
    data.dropna(inplace=True)
    X = data.drop(columns=['target'])
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    pipelines = {
        'lr': Pipeline([('clf', SklearnLogisticRegression())]),
    }

    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    scores = {}
    for model_name, pipeline in pipelines.items():
        scores[model_name] = {}
        for scorer_name, scorer in scorers.items():
            score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
            scores[model_name][scorer_name] = score.mean()

    kwargs['ti'].xcom_push(key='sklearn_lr_accuracy', value=scores['lr']['accuracy'])

def sklearn_svm(**kwargs):
    data = pd.read_csv(fe_sklearn_path)
    data.dropna(inplace=True)
    X = data.drop(columns=['target'])
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    pipelines = {
        'svc': Pipeline([('clf', SVC())]),
    }

    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score)
    }

    scores = {}
    for model_name, pipeline in pipelines.items():
        scores[model_name] = {}
        for scorer_name, scorer in scorers.items():
            score = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=scorer)
            scores[model_name][scorer_name] = score.mean()

    kwargs['ti'].xcom_push(key='sklearn_svc_accuracy', value=scores['svc']['accuracy'])

# Function to train and evaluate models using Spark
def spark_lr(**kwargs):
    spark = create_spark_session()
    df = spark.read.csv(fe_spark_path, header=True, inferSchema=True)

    df = df.dropna()

    # Ensure 'target' column is categorical
    indexer = StringIndexer(inputCol="target", outputCol="label")
    df = indexer.fit(df).transform(df)

    # Define the feature columns
    feature_cols = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope']

    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast('double'))

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df = assembler.transform(df)

    # Split the data into training and test sets with 90-10 split and stratification
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)

    # Verify the splits
    train_data.groupBy("label").count().show()
    test_data.groupBy("label").count().show()

    # Train Logistic Regression model
    lr = LogisticRegression(labelCol='target', featuresCol='features', maxIter=10)
    lr_model = lr.fit(df)
    
    # Evaluate models
    evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    lr_accuracy = evaluator.evaluate(lr_model.transform(df))
    
    kwargs['ti'].xcom_push(key='spark_lr_accuracy', value=lr_accuracy)

def spark_svm(**kwargs):
    spark = create_spark_session()
    df = spark.read.csv(fe_spark_path, header=True, inferSchema=True)

    df = df.dropna()

    # Ensure 'target' column is categorical
    indexer = StringIndexer(inputCol="target", outputCol="label")
    df = indexer.fit(df).transform(df)

    # Define the feature columns
    feature_cols = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope']

    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast('double'))

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df = assembler.transform(df)

    # Split the data into training and test sets with 90-10 split and stratification
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)

    # Verify the splits
    train_data.groupBy("label").count().show()
    test_data.groupBy("label").count().show()
    
    # Train Support Vector Classifier (SVC) model
    svc = LinearSVC(labelCol='target', featuresCol='features', maxIter=10)
    svc_model = svc.fit(df)
    
    # Evaluate models
    evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    svc_accuracy = evaluator.evaluate(svc_model.transform(df))
    
    kwargs['ti'].xcom_push(key='spark_svc_accuracy', value=svc_accuracy)

def merge_lr(**kwargs):
    spark = create_spark_session()
    df = spark.read.csv(merged_path, header=True, inferSchema=True)

    df = df.dropna()

    # Ensure 'target' column is categorical
    indexer = StringIndexer(inputCol="target", outputCol="label")
    df = indexer.fit(df).transform(df)

    # Define the feature columns
    feature_cols = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope']

    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast('double'))

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df = assembler.transform(df)

    # Split the data into training and test sets with 90-10 split and stratification
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)

    # Verify the splits
    train_data.groupBy("label").count().show()
    test_data.groupBy("label").count().show()

    # Train Logistic Regression model
    lr = LogisticRegression(labelCol='target', featuresCol='features', maxIter=10)
    lr_model = lr.fit(df)
    
    # Evaluate models
    evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    lr_accuracy = evaluator.evaluate(lr_model.transform(df))
    
    kwargs['ti'].xcom_push(key='merge_lr_accuracy', value=lr_accuracy)

def merge_svm(**kwargs):
    spark = create_spark_session()
    df = spark.read.csv(merged_path, header=True, inferSchema=True)

    df = df.dropna()

    # Ensure 'target' column is categorical
    indexer = StringIndexer(inputCol="target", outputCol="label")
    df = indexer.fit(df).transform(df)

    # Define the feature columns
    feature_cols = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope']

    for col_name in feature_cols:
        df = df.withColumn(col_name, col(col_name).cast('double'))

    # Assemble feature columns into a single vector column
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    df = assembler.transform(df)

    # Split the data into training and test sets with 90-10 split and stratification
    train_data, test_data = df.randomSplit([0.9, 0.1], seed=42)

    # Verify the splits
    train_data.groupBy("label").count().show()
    test_data.groupBy("label").count().show()
    
    # Train Support Vector Classifier (SVC) model
    svc = LinearSVC(labelCol='target', featuresCol='features', maxIter=10)
    svc_model = svc.fit(df)
    
    # Evaluate models
    evaluator = MulticlassClassificationEvaluator(labelCol='target', predictionCol='prediction', metricName='accuracy')
    svc_accuracy = evaluator.evaluate(svc_model.transform(df))
    
    kwargs['ti'].xcom_push(key='merge_svc_accuracy', value=svc_accuracy)

def choose_best_model(**kwargs):
    # Accessing the accuracy scores stored in XCOM
    sklearn_lr_accuracy = kwargs['ti'].xcom_pull(task_ids='sklearn_lr', key='sklearn_lr_accuracy')
    sklearn_rf_accuracy = kwargs['ti'].xcom_pull(task_ids='sklearn_svm', key='sklearn_svc_accuracy')
    spark_lr_accuracy = kwargs['ti'].xcom_pull(task_ids='spark_lr', key='spark_lr_accuracy')
    spark_svc_accuracy = kwargs['ti'].xcom_pull(task_ids='spark_svm', key='spark_svc_accuracy')
    merge_lr_accuracy = kwargs['ti'].xcom_pull(task_ids='merge_lr', key='merge_lr_accuracy')
    merge_svc_accuracy = kwargs['ti'].xcom_pull(task_ids='merge_svm', key='merge_svc_accuracy')

    # Comparing accuracy scores
    model_accuracies = {
        'sklearn_lr': sklearn_lr_accuracy,
        'sklearn_rf': sklearn_rf_accuracy,
        'spark_lr': spark_lr_accuracy,
        'spark_svc': spark_svc_accuracy,
        'merge_lr': merge_lr_accuracy,
        'merge_svc': merge_svc_accuracy
    }

    best_model = max(model_accuracies, key=model_accuracies.get)

    print("The best model is:", best_model)

# load data
load_data_s3 = PythonOperator(
    task_id='load_data_s3',
    python_callable=load_data_s3,
    dag=dag,
)

# clean data
sklearn_clean_data_task = PythonOperator(
    task_id='sklearn_clean_and_impute_data',
    python_callable=sklearn_clean_and_impute_data,
    provide_context=True,
    dag=dag,
)
spark_clean_data_task = PythonOperator(
    task_id='spark_clean_and_impute_data',
    python_callable=spark_clean_and_impute_data,
    provide_context=True,
    dag=dag,
)

# fe
fe_sklearn = PythonOperator(
    task_id='fe_sklearn',
    python_callable=fe_sklearn,
    provide_context=True,
    dag=dag,
)
fe_spark = PythonOperator(
    task_id='fe_spark',
    python_callable=fe_spark,
    provide_context=True,
    dag=dag,
)

# web scraping
scrape = PythonOperator(
    task_id='scrape',
    python_callable=scrape,
    provide_context=True,
    dag=dag,
)

# merging
merge = PythonOperator(
    task_id='merge',
    python_callable=merge,
    provide_context=True,
    dag=dag,
)

# train and choose models
sklearn_lr = PythonOperator(
    task_id='sklearn_lr',
    python_callable=sklearn_lr,
    dag=dag,
)
sklearn_svm = PythonOperator(
    task_id='sklearn_svm',
    python_callable=sklearn_svm,
    dag=dag,
)
spark_lr = PythonOperator(
    task_id='spark_lr',
    python_callable=spark_lr,
    dag=dag,
)
spark_svm = PythonOperator(
    task_id='spark_svm',
    python_callable=spark_svm,
    dag=dag,
)
merge_lr = PythonOperator(
    task_id='merge_lr',
    python_callable=merge_lr,
    dag=dag,
)
merge_svm = PythonOperator(
    task_id='merge_svm',
    python_callable=merge_svm,
    dag=dag,
)

# choose best model
best_model = PythonOperator(
    task_id='choose_best_model',
    python_callable=choose_best_model,
    provide_context=True,
    dag=dag
)

load_data_s3 >> [sklearn_clean_data_task, spark_clean_data_task] 
sklearn_clean_data_task >> fe_sklearn 
spark_clean_data_task >> [fe_spark, scrape]
fe_sklearn >> [sklearn_lr, sklearn_svm]
fe_spark >> [spark_lr, spark_svm]

[fe_sklearn, fe_spark, scrape] >> merge
merge >> [merge_lr, merge_svm]
[sklearn_lr, sklearn_svm, spark_lr, spark_svm, merge_lr, merge_svm] >> best_model
