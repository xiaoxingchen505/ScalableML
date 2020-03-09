import pyspark

from pyspark.sql import SparkSession
from pyspark.sql import Row
import pyspark.sql.functions as func
import matplotlib.pyplot as plt


spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment 1 Task1") \
    .getOrCreate()

sc = spark.sparkContext

logFile=spark.read.text("NASA_access_log_Jul95.gz").cache()

reqs = logFile.withColumn('time', func.regexp_extract(logFile['value'],':(.*) -',1))
reqs_with_data = reqs.withColumn('Date', func.regexp_extract(func.col('value'),'\[(\d+)\/Jul',1))
df_first_slot = reqs_with_data.filter(func.col("time").between("00:00:00","03:59:59"))
avg_1=df_first_slot.groupBy('Date').count().select(func.mean('count')).collect()[0]
df_second_slot = reqs_with_data.filter(func.col("time").between("04:00:00","07:59:59"))
avg_2=df_second_slot.groupBy('Date').count().select(func.mean('count')).collect()[0]
df_third_slot = reqs_with_data.filter(func.col("time").between("08:00:00","11:59:59"))
avg_3=df_third_slot.groupBy('Date').count().select(func.mean('count')).collect()[0]
df_fourth_slot = reqs_with_data.filter(func.col("time").between("12:00:00","15:59:59"))
avg_4=df_fourth_slot.groupBy('Date').count().select(func.mean('count')).collect()[0]
df_fifth_slot = reqs_with_data.filter(func.col("time").between("16:00:00","19:59:59"))
avg_5=df_fifth_slot.groupBy('Date').count().select(func.mean('count')).collect()[0]
df_sixth_slot = reqs_with_data.filter(func.col("time").between("20:00:00","23:59:59"))
avg_6=df_fifth_slot.groupBy('Date').count().select(func.mean('count')).collect()[0]

rdd_report = sc.parallelize([('0 -- 4',avg_1),
                             ('4 -- 8',avg_2),
                             ('8 -- 12',avg_3),
                             ('12 -- 16',avg_4),
                             ('16 -- 20',avg_5),
                             ('20 -- 24',avg_6)])
df_report = rdd_report.toDF(["Six different Timeslot","Average number of request of Jul"])
df_report.show()

import numpy as np
df_pd = df_report.toPandas()


plt.bar(df_pd['Six different Timeslot'],df_pd['Average number of request of Jul'],align='edge',width =0.6)
plt.title("Requests of each time slot")
plt.xlabel("Time slot")
plt.ylabel("Request Numbers")
#plt.savefig('Q1A_FIG1')

reqs_file = logFile.withColumn('File_Name', func.regexp_extract(logFile['value'],'(?s:.*)\/(.*) HTTP',1)

df_new = reqs_file.filter(reqs_file['File_Name'].contains(".html"))

reqs_df_count=df_new.groupBy('File_Name').count().sort('count', ascending=False)
reqs_df_count = reqs_df_count.limit(20)
reqs_df_count.show(20,False)

plt.barh(df_reqs['File_Name'],height =0.5,width = df_reqs['count'])
plt.title("Top 20 Requested .html File")
plt.xlabel("Number of Request")
plt.ylabel("Requested .html file")
plt.show()