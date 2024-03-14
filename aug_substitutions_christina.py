# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Clickstream Data for Carside Accept/Reject Substitutions
# MAGIC Work to support substitutes science for e-commerce orders. 
# MAGIC
# MAGIC Context: *If the ordered item is out of stock, the science recommends appropriate substitute products. Training data is currently limited to customers who opt into two-way communication text messages, which allow customers to accept/reject subs before picking up their order. Missing data from customers who accept/reject their subs "carside" at time of pickup.*
# MAGIC
# MAGIC Question: **The primary objective is to identify whether clickstream data contains the data points regarding carside accept/reject data.**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC *Why is ACDS transaction data to confirm "accept/reject" not enough?*
# MAGIC
# MAGIC * If we find the item in the transaction, we don't know if it's there because the sub was accepted or because the customer also purchased that item. It's possible they purchased that item separately, but also rejected it as a substitute for the other item. But it would look like an accept when joining harvester and acds transactions.
# MAGIC * Some customers blindly accept or reject all of their subs, without thinking about each one individually. This doesn't give us much useful information, so we typically the exclude orders where all or none of the harvester records are found in acds transactions. This way we focus on "engaged" customers, who actively considered each sub. (There are obviously some large assumptions about what's happening with these customers, but we've found that when we used this approach the acceptance rate correlates better with the 2WC acceptance rate.)
# MAGIC * So yes, we can use ACDS transactions to estimate whether subs were accepted or rejected, but we always prefer an explicit data point from the customer, like 2WC.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Import Packages

# COMMAND ----------

# access substitution datasets
# here will use (harvester) substitution events
# also substitution response events aka 2WC, explicit accept/reject text data

%pip install effosubs

# COMMAND ----------

import os
from functools import reduce

import effosubs

from effodata import ACDS

from pyspark.sql import functions as f

acds = ACDS(use_sample_mart=False)

# COMMAND ----------

# date selection
# matching dates on pulled clickstream data in directory
START_DATE = "2023-08-01"
END_DATE = "2023-08-31"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clickstream Data

# COMMAND ----------

directory = "abfss://sandbox@sa8451learningdev.dfs.core.windows.net/users/c744990/project_pool"

clicks_path = os.path.join(directory, "clicks_sample_aug_2023")
product_path = os.path.join(directory, "product_table_sample_aug_2023")

# read clickstream files
clicks_df = spark.read.parquet(clicks_path, header=True, inferSchema=True)
clicks_product_df = spark.read.parquet(product_path, header=True, inferSchema=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Does 'effo_channel' indicate a device used for carside data?
# MAGIC No, (for August 2023 data) 'effo_channel' only indicates user activity in the app or in a web browser. There is no 'effo_channel' unique to the harvester device/carside pickup.

# COMMAND ----------

# check whether there is anything interesting with device used
clicks_df.select("effo_channel").distinct().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### How many records in the clickstream data include accepted/rejected information?
# MAGIC For Augus 2023 data, 50.6% of clickstream data includes accepted/rejected information.

# COMMAND ----------

# selecting clickstream columns of interest
sub_cols = [col for col in clicks_df.columns if 'sub' in col]
sub_cols.append('upc')
sub_cols.append('order_id')
sub_cols.append('clickstream_accepted_rejected_record')
# sub_cols.append('effo_channel') # don't include, not interesting

clicks_df_subs = (
  clicks_df  
  .withColumn('clickstream_accepted_rejected_record',
              f.when(f.col('sub_status').isNull(), f.lit('NO'))
              .otherwise(f.lit('YES')))
  .select(sub_cols)
)

# checking counts of how many records we have accepted/rejected info
(clicks_df_subs.groupBy('clickstream_accepted_rejected_record')
.agg(f.count('clickstream_accepted_rejected_record').alias('count'), 
     ((f.count('clickstream_accepted_rejected_record')/clicks_df_subs.count())*100).alias('percent'))
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Format 'sub_upc' column
# MAGIC
# MAGIC * sub_upc are recorded as arrays, some with muliple values
# MAGIC * use explode to create a row for each element in each sub_upc list
# MAGIC * then, drop distinct to include only unique combinations of order_id, upc, sub_upc
# MAGIC
# MAGIC Ultimately will join on Harvester data on: order_id=order_number, upc=order_upc, sub_upc=picked_upc

# COMMAND ----------

# sub_upc are recorded as arrays, some with muliple values
# use explode to create a row for each element in each sub_upc list
# then, drop distinct to include only unique combinations of order_id, upc, sub_upc
# ultimately will join on Harvester data on order_id=order_number, upc=order_upc, sub_upc=picked_upc

clickstream_df = (clicks_df_subs
                  .withColumn('sub_upc', f.explode(clicks_df_subs.sub_upc))
                  .dropDuplicates()
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Substitution Data

# COMMAND ----------

# get substitution data from effosubs

# data from kroger associate device
# history of all substitutions that occured
# recorded before customer accepts/rejects, does not include this info
harvester_df = effosubs.get_substitution_events(start_date=START_DATE, end_date=END_DATE)

# history of 2-way comms text messages
# includes accept/reject data *only* from customers that responded to text messages
customer_response_df_with_ocado = effosubs.get_substitution_response_events(start_date=START_DATE, end_date=END_DATE)

# some customer response data is from orders that were placed for pickup
# actually fulfilled with ocado and still labeled with store div no
# use this to filter it out
ocado_with_subs_df = effosubs.get_ocado_fulfillment_events(start_date=START_DATE, end_date=END_DATE, only_include_subs=True)

# filter out rows with ocado denoted by division 540
customer_response_df = (customer_response_df_with_ocado
                .where(f.col('division') != 540)
                .withColumnRenamed('store_code','cust_resp_store_code')
                )

harvester_df.count()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Join Harvester data with customer response data
# MAGIC * Main goal is to identify records in Harvester without Customer Response data
# MAGIC * There are customer response records that are not in Harvester, only partially explained by Ocado events
# MAGIC   * Explanation is still unclear. TODO

# COMMAND ----------

# MAGIC %md
# MAGIC #### Unmatched Harvester / Customer Response Data
# MAGIC
# MAGIC * There are entries in 2WC, customer_response_df (text accept/reject data) that are not being matched with Harvester data. 
# MAGIC * Impression was that every instance in customer_response_df should have a match in harvester_df (then we look to clickstream to find complement).
# MAGIC * 7.08% of customer_response_df records with NO harvester data
# MAGIC * 3.89% of customer_response_df records that are ocado events still in dataframe.
# MAGIC
# MAGIC
# MAGIC * harvester_df count: 3,131,002 (August 2023)
# MAGIC   * customer_response_df count: 2,164,096 (August 2023)
# MAGIC   * inner join on ['order_no', 'ordered_upc', 'picked_upc']: 2,010,794
# MAGIC   * Harvester records without Customer Response 2WC data: 1,120,208

# COMMAND ----------

# joining general customer response data and customer response data specific to ocado sub events
ocado_innerjoin_customer_response = (
  customer_response_df.join(other=ocado_with_subs_df, on=['order_no', 'ordered_upc', 'picked_upc'], how='inner')
)

ocado_innerjoin_customer_response.count()

# COMMAND ----------

# joining harvester and customer response data
# looking for unmatched harvester records that may be covered with clickstream
# also notice customer response records that are not in harvester data 

innerjoin_harvester_response = (
  harvester_df.join(other=customer_response_df, on=['order_no', 'ordered_upc', 'picked_upc'], how='inner')
)

leftjoin_harvester_response = (
  harvester_df.join(other=customer_response_df, on=['order_no', 'ordered_upc', 'picked_upc'], how='left')
)

rightjoin_harvester_response = (
  harvester_df.join(other=customer_response_df, on=['order_no', 'ordered_upc', 'picked_upc'], how='right')
)

# COMMAND ----------

# counting entries without accept/reject info in join_harvester_response_df
# this difference indicates that there are entries in the customer_response_df that are not matching with entries in harvester_df

print(f"{innerjoin_harvester_response.count()} records in (inner) joined harvester/customer_response dataframe") 

# use null values in attempted_utilization column because this is only in harvester_df
no_harvester = rightjoin_harvester_response.where(rightjoin_harvester_response.attempted_utilization.isNull())
print(f"{round((no_harvester.count()/customer_response_df.count())*100,2 )}% of customer_response_df records with NO harvester data")

print(f"{round((ocado_innerjoin_customer_response.count()/customer_response_df.count())*100, 2)}% of customer_response_df records that are ocado events still in dataframe.")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Harvester Records without Customer Response
# MAGIC
# MAGIC 35.8% of harvester data **do not** have 2WC record of accept/reject

# COMMAND ----------

harvester_with_2wc_flag = (
  leftjoin_harvester_response
  .withColumn('2Wc_accept_reject_record',
              f.when(f.col('status').isNull(), f.lit('NO'))
              .otherwise(f.lit('YES')))
  .select(['order_no', 'ordered_upc', 'picked_upc','status','2Wc_accept_reject_record','store_code'])
)

# checking counts
(harvester_with_2wc_flag.groupBy('2Wc_accept_reject_record')
 .agg(f.count('2Wc_accept_reject_record').alias('count'), 
     ((f.count('2Wc_accept_reject_record')/harvester_df.count())*100).alias('percent'))
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Join Clickstream with Harvester

# COMMAND ----------

# all columns with sub/substitute in name are from clickstream
# reorder columns 
focus_ordered_columns = [
  '2Wc_accept_reject_record',
  'clickstream_accepted_rejected_record',
  'order_no', 
  'ordered_upc',
  'picked_upc', 
  'sub_upc', 
  'status',
  'sub_status',
  'store_code',
  'allow_substitutes',
 'number_of_subs',
 'substitute_method',
 'product_allow_substitutes',
 'product_substitute_method',
 'sub_name',
 'sub_price',
 'sub_type',
 'sub_units']

harvester_with_clicks = (
  harvester_with_2wc_flag.join(other=clickstream_df, on=[harvester_with_2wc_flag.order_no==clickstream_df.order_id, harvester_with_2wc_flag.ordered_upc==clickstream_df.upc, harvester_with_2wc_flag.picked_upc==clickstream_df.sub_upc], how='left')
  .select(focus_ordered_columns)
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Minimal opportunity to recover missing response data in Clickstream
# MAGIC Less that 1% of February 2024 missing response data has a response in Clickstream.

# COMMAND ----------

# how many records without 2WC info do we have clickstream data for in this way

no_2wc = (harvester_with_clicks.where(f.col('2Wc_accept_reject_record') == 'NO'))
no_2wc_yes_clicks = (harvester_with_clicks
                    .where((f.col('2Wc_accept_reject_record') == 'NO') 
                          & (f.col('clickstream_accepted_rejected_record') == 'YES'))
)

print(f"{round((no_2wc_yes_clicks.count()/no_2wc.count())*100,2)}% of Harvester records missing accept/reject 2WC information that have accepted/rejected information in Clickstream data.")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### **Contradictions** in response data: clickstream vs. effo_subs (2WC)
# MAGIC
# MAGIC Unexpected discovery: (August 2023) harvester (2WC) records with accept/reject data that **do not** match accepted/rejected data in clickstream.
# MAGIC
# MAGIC For August 2023:
# MAGIC * 15.72% of all 2WC customer response ('status') records do NOT match with clickstream 'sub_status'
# MAGIC * 314,898 records, which is 99.99% of 2WC records that do NOT match with clickstream 'sub_status', are recorded as 2WC 'accept' (and as 'rejected' in clicktream)
# MAGIC * 30 records, which is 0.01% of 2WC records that do NOT match with clickstream 'sub_status', are recorded as 2WC 'reject' (and as 'accepted') in clickstream

# COMMAND ----------

# check how accept/reject is indicated in 2WC data and clickstream data
# first check values to look for

# 2WC accept/reject
harvester_with_clicks.select("status").distinct().show()

# clickstream accept/reject
harvester_with_clicks.select("sub_status").distinct().show()

# COMMAND ----------

yes_2wc_yes_clicks = (harvester_with_clicks
                      .where((f.col('2Wc_accept_reject_record') == 'YES') 
                          & (f.col('clickstream_accepted_rejected_record') == 'YES'))
)


# filter for 2WC Accept/Reject does NOT match with clickstream accepted/rejected
count_non_match = (
  yes_2wc_yes_clicks.where(((f.col('status') == 'Accept') 
                          & (f.col('sub_status') == 'rejected'))
                           |
                           ((f.col('status') == 'Reject') 
                          & (f.col('sub_status') == 'accepted'))
                        )
).count()

print(f"{round((count_non_match/yes_2wc_yes_clicks.count())*100,2)}% of all 2WC customer response ('status') records do NOT match with clickstream 'sub_status'")

# Out of mismatched responses, what percent are 'accept' by 2WC
count_accept_mismatch = (
  yes_2wc_yes_clicks.where((f.col('status') == 'Accept') 
                          & (f.col('sub_status') == 'rejected')
                        )
).count()

print(f"{count_accept_mismatch} records, {round((count_accept_mismatch/count_non_match)*100,2)}% of 2WC records that do NOT match with clickstream 'sub_status' are recorded as 2WC 'accept' (and as 'rejected' in clicktream)")

# Out of mismatched responses, what percent are 'reject' by 2WC
count_reject_mismatch = (
  yes_2wc_yes_clicks.where((f.col('status') == 'Reject') 
                          & (f.col('sub_status') == 'accepted')
                        )
).count()

print(f"{count_reject_mismatch} records, {round((count_reject_mismatch/count_non_match)*100,2)}% of 2WC records that do NOT match with clickstream 'sub_status' are recorded as 2WC 'reject' (and as 'accepted') in clickstream")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Join with ACDS
# MAGIC On an inner join with transaction data mapping picked_upc (from effo subs data) to gtin_no, we have all transactions accounted for. This matched count leads to the conclusion that effo data accept/reject record is more reliable than the clickstream accepted/rejected record.

# COMMAND ----------

# look at records with mismatched responses
# note: all price difference are -1.0
non_match_df = (
  yes_2wc_yes_clicks.where(((f.col('status') == 'Accept') 
                          & (f.col('sub_status') == 'rejected'))
                           |
                           ((f.col('status') == 'Reject') 
                          & (f.col('sub_status') == 'accepted'))
                        )
)

non_match_df.count()

# COMMAND ----------

# look at transactions
txns = acds.get_transactions(
  start_date=START_DATE,
  end_date=END_DATE,
)

# txns.display()

# COMMAND ----------

# joined records with mismtached accept/reject to transaction data
non_match_response_join_acds = (
  non_match_df.join(other=txns, on=[
    non_match_df.order_no==txns.order_no, 
    non_match_df.picked_upc==txns.gtin_no, 
    non_match_df.store_code==txns.store_code], how='inner')
)

# COMMAND ----------

# confirm count is same as non_match_df
non_match_response_join_acds.count()
