import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row,Column
import pandas as pd
import numpy as np
import time

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment Task2") \
    .config("spark.driver.memory", "50g")\
    .config('spark.local.dir','/fastdata/acq18xx')\
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("OFF")

sc.setCheckpointDir('checkpoint/')

#start Data pre-processing
raw_df = spark.read.csv('ClaimPredictionChallenge/train_set.csv',header=True)

raw_df = raw_df.select('Vehicle','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1','Cat2','Cat3','Cat4','Cat5','Cat6','Cat7','Cat8','Cat9','Cat10','Cat11','Cat12','Calendar_Year','Model_Year','Claim_Amount')

print('')
print('Start data preprocessing...')
print('')
for col in raw_df.columns:
    raw_df = raw_df.filter((raw_df[col] != '?'))

from pyspark.sql.types import DoubleType
from pyspark.sql.types import IntegerType

double_data = ['Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4']
int_data = ['Vehicle','Calendar_Year','Model_Year','Claim_Amount']
input_features = double_data+int_data
for col in double_data:
    raw_df = raw_df.withColumn(col, raw_df[col].cast(DoubleType()))
for col in int_data:
    raw_df = raw_df.withColumn(col, raw_df[col].cast(IntegerType()))

raw_df = raw_df.withColumn(int_data[0], (raw_df[int_data[0]]-2005))
raw_df = raw_df.withColumn(int_data[1], (raw_df[int_data[1]]-1981))

#categorical Nbr Lvls in Train for each cat
categorical_features = {'Cat1':11,'Cat2':4,'Cat3':7,'Cat4':4,'Cat5':4,'Cat6':7,'Cat7':5,'Cat8':4,'Cat9':2,'Cat10':4,'Cat11':7,'Cat12':7}


from pyspark.ml.feature import StringIndexer,OneHotEncoderEstimator

for col,num in categorical_features.items():
    name = col+'_id'
    indexer = StringIndexer(inputCol=col, outputCol=name)
    raw_df = indexer.fit(raw_df).transform(raw_df)

raw_df = raw_df.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8',\
                        'NVVar1','NVVar2','NVVar3','NVVar4',\
                        'Cat1_id','Cat2_id','Cat3_id','Cat4_id','Cat5_id','Cat6_id','Cat7_id','Cat8_id','Cat9_id','Cat10_id','Cat11_id','Cat12_id',\
                        'Calendar_Year','Model_Year','Claim_Amount')

category_id = ['Cat1_id','Cat2_id','Cat3_id','Cat4_id','Cat5_id','Cat6_id','Cat7_id','Cat8_id','Cat9_id','Cat10_id','Cat11_id','Cat12_id']
cat_ohe = []
for col in category_id:
    cat_ = col.replace('_id','_ohe')
    cat_ohe.append(cat_)
    input_features.append(cat_)


data = raw_df.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_id','Cat2_id','Cat3_id','Cat4_id','Cat5_id','Cat6_id','Cat7_id','Cat8_id','Cat9_id','Cat10_id','Cat11_id','Cat12_id','Calendar_Year','Model_Year','Claim_Amount')

encoder = OneHotEncoderEstimator(inputCols=category_id, outputCols=cat_ohe)

encoder_data = encoder.fit(data)
data_ohe = encoder_data.transform(data)


# #assemble all features
from pyspark.ml.feature import VectorAssembler
all_features_assembler = VectorAssembler(inputCols=['Cat1_ohe','Cat2_ohe','Cat3_ohe','Cat4_ohe'\
                                                    ,'Cat5_ohe','Cat6_ohe','Cat7_ohe','Cat8_ohe'\
                                                    ,'Cat9_ohe','Cat10_ohe','Cat11_ohe','Cat12_ohe'\
                                                    ,'Var1','Var2','Var3','Var4','Var5',\
                                                        'Var6','Var7','Var8','NVVar1','NVVar2'\
                                                    ,'NVVar3','NVVar4','Calendar_Year','Model_Year'],\
                                                    outputCol='features')
all_data = all_features_assembler.transform(data_ohe)


#start to train model
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when

t_data= all_data.select('features','Claim_Amount')
(trainingData, testData) = t_data.randomSplit([0.7, 0.3], 47)


trainingData.cache()  
testData.cache()
print('Data preprocessing finished.')

lr = LinearRegression(featuresCol="features", labelCol="Claim_Amount")


import time
start = time.time()
print('Start training......')
lr_Model = lr.fit(trainingData)

lr_prediction = lr_Model.transform(testData)

evaluator = RegressionEvaluator(labelCol="Claim_Amount",\
                                predictionCol="prediction",\
                                metricName="rmse")

lr_rmse = evaluator.evaluate(lr_prediction)
end = time.time()
print('')
print('Execution time:',end-start)
print('')
print("RMSE = %g" % lr_rmse)
print('')
evaluator = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")
lr_mae = evaluator.evaluate(lr_prediction)
print("MAE = %g" % lr_mae)
