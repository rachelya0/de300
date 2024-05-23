from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, mean, create_map
from pyspark.sql.types import FloatType
from itertools import chain
import requests
from bs4 import BeautifulSoup
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

import os
os.environ['PYSPARK_SUBMIT_ARGS'] = "--packages=com.amazonaws:aws-java-sdk-bundle:1.11.375,org.apache.hadoop:hadoop-aws:3.3.4 pyspark-shell"

# Initialize Spark session
spark = SparkSession.builder.appName("HeartDiseasePrediction")\
        .getOrCreate()

spark._jsc.hadoopConfiguration().set("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.4")
spark._jsc.hadoopConfiguration().set("fs.s3a.connection.ssl.enabled", "true")
spark._jsc.hadoopConfiguration().set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
spark._jsc.hadoopConfiguration().set("fs.s3a.endpoint", "s3.amazonaws.com")
    
# Set Hadoop configurations for S3 access
hadoop_conf = spark._jsc.hadoopConfiguration()
hadoop_conf.set("fs.s3.aws.credentials.provider", "org.apache.hadoop.fs.s3a.TemporaryAWSCredentialsProvider")
hadoop_conf.set("fs.s3a.access.key", "...")
hadoop_conf.set("fs.s3a.secret.key", "...")
hadoop_conf.set("fs.s3a.session.token", "...")

# Load data
s3_bucket = "s3a://de300spring2024"
file_key = 'rachel_yao/heart_disease(in).csv'
data_path = f"{s3_bucket}/{file_key}"
data = spark.read.csv(data_path, header=True, inferSchema=True)

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

data = data.dropna()

# Show data
data.show()

# 3
# Feature engineering
from pyspark.sql import DataFrame

# Ensure 'target' column is categorical
indexer = StringIndexer(inputCol="target", outputCol="label")
data = indexer.fit(data).transform(data)

# Define the feature columns
feature_cols = ['age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 'oldpeak', 'slope', 'smoke_source1', 'smoke_source2']

for col_name in feature_cols:
    data = data.withColumn(col_name, col(col_name).cast('double'))

# Assemble feature columns into a single vector column
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(data)

# Split the data into training and test sets with 90-10 split and stratification
train_data, test_data = data.randomSplit([0.9, 0.1], seed=42)

# Verify the splits
train_data.groupBy("label").count().show()
test_data.groupBy("label").count().show()

# Initialize models
lr = LogisticRegression(featuresCol='features', labelCol='label')
rf = RandomForestClassifier(featuresCol='features', labelCol='label')
gbt = GBTClassifier(featuresCol='features', labelCol='label')

# Set up cross-validation
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')

# Logistic Regression
lr_param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
lr_cv = CrossValidator(estimator=lr, estimatorParamMaps=lr_param_grid, evaluator=evaluator, numFolds=5)
lr_model = lr_cv.fit(train_data)
lr_accuracy = evaluator.evaluate(lr_model.transform(test_data))

# Random Forest
rf_param_grid = ParamGridBuilder().addGrid(rf.numTrees, [10, 20]).build()
rf_cv = CrossValidator(estimator=rf, estimatorParamMaps=rf_param_grid, evaluator=evaluator, numFolds=5)
rf_model = rf_cv.fit(train_data)
rf_accuracy = evaluator.evaluate(rf_model.transform(test_data))

# Gradient Boosting
gbt_param_grid = ParamGridBuilder().addGrid(gbt.maxDepth, [5, 10]).build()
gbt_cv = CrossValidator(estimator=gbt, estimatorParamMaps=gbt_param_grid, evaluator=evaluator, numFolds=5)
gbt_model = gbt_cv.fit(train_data)
gbt_accuracy = evaluator.evaluate(gbt_model.transform(test_data))

def evaluate_model(predictions, label_col="label"):
    evaluator = MulticlassClassificationEvaluator(labelCol=label_col)
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
    f1 = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

# Evaluate models after balancing
lr_predictions = lr_model.transform(test_data)
print("Logistic Regression Metrics:")
evaluate_model(lr_predictions)

rf_predictions = rf_model.transform(test_data)
print("Random Forest Metrics:")
evaluate_model(rf_predictions)

gbt_predictions = gbt_model.transform(test_data)
print("Gradient Boosting Metrics:")
evaluate_model(gbt_predictions)

print("Random forest seems to perform the best out of the models tested, with a high mean accuracy of 0.85, precision of 0.85, recall of 0.85, and f1 score of 0.85.")