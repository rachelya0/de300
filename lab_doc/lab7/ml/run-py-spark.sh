# export PYSPARK_DRIVER_PYTHON=python3 only in cluster mode
export PYSPARK_PYTHON=../demos/bin/python3
/opt/spark/bin/spark-submit --archives ../demos.tar.gz#demos ml_pyspark.ipynb
