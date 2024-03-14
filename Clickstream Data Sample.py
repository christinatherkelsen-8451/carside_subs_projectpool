# Databricks notebook source
# MAGIC %pip install effodata_clickstream

# COMMAND ----------

import os
from functools import reduce
from effodata_clickstream.effodata_clickstream import CSDM
from pyspark.sql import functions as F

import pandas as pd
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.max_colwidth", 50)

# COMMAND ----------

START_DATE = "2023-08-01"
END_DATE = "2023-08-31"

OUTPUT_DIR = "abfss://sandboax@sa8451learningdev.dfs.core.windows.net/users/c744990/project_pool"

# COMMAND ----------

csdm = CSDM(spark)

# COMMAND ----------

# pull clickstream fact table
# join with product dataset
# filter to only include data with subs 

df = csdm.get_clickstream_records(
    start_date=START_DATE,
    end_date=END_DATE,
    join_with=["product"],
    customer_behavior_only=False,  # probably don't need to set explicitly because the default is False
)

# COMMAND ----------


