#!/bin/bash

# Path to your python script
PYTHON_SCRIPT="classify.py"

# Submit the PySpark job to EMR
spark-submit --master yarn --deploy-mode cluster $PYTHON_SCRIPT

