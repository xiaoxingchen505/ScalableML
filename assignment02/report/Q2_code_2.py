import pyspark
from pyspark.sql import SparkSession
from pyspark.sql import Row,Column
import pandas as pd
import numpy as np
import time
from pyspark.sql.functions import when,log,exp
from pyspark.ml.feature import StringIndexer,OneHotEncoderEstimator

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Assignment Task2-2") \
        .config("spark.driver.memory", "50g")\
	.config('spark.local.dir','/fastdata/acq18xx')\
    .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("OFF")

sc.setCheckpointDir('checkpoint/')

#start Data pre-processing

print('')
print('Start data preprocessing...')
print('')

raw_df = spark.read.csv('ClaimPredictionChallenge/train_set.csv',header= True)

raw_df =raw_df.select('Vehicle','Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1','Cat2','Cat3','Cat4','Cat5','Cat6','Cat7','Cat8','Cat9','Cat10','Cat11','Cat12','Calendar_Year','Model_Year','Claim_Amount')

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


raw_df = raw_df.withColumn(int_data[1], (raw_df[int_data[1]]-2005))
raw_df = raw_df.withColumn(int_data[2], (raw_df[int_data[2]]-1981))

#categorical Nbr Lvls in Train for each cat
categorical_features = {'Cat1':11,'Cat2':4,'Cat3':7,'Cat4':4,'Cat5':4,'Cat6':7,'Cat7':5,'Cat8':4,'Cat9':2,'Cat10':4,'Cat11':7,'Cat12':7}

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
data  = encoder_data.transform(data)


#assign weight on data
data  = data .withColumn('weight',when((data['Claim_Amount'] != 0), 0.98).otherwise(0.02))
data  = data .withColumn('not_zero',when((data['Claim_Amount'] != 0), 1).otherwise(0))


data = data.select('Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_ohe','Cat2_ohe','Cat3_ohe','Cat4_ohe','Cat5_ohe','Cat6_ohe','Cat7_ohe','Cat8_ohe','Cat9_ohe','Cat10_ohe','Cat11_ohe','Cat12_ohe','Calendar_Year','Model_Year','Claim_Amount','weight','not_zero')


features_list = ['Var1','Var2','Var3','Var4','Var5','Var6','Var7','Var8','NVVar1','NVVar2','NVVar3','NVVar4',
                  'Cat1_ohe','Cat2_ohe','Cat3_ohe','Cat4_ohe','Cat5_ohe','Cat6_ohe','Cat7_ohe','Cat8_ohe','Cat9_ohe','Cat10_ohe','Cat11_ohe','Cat12_ohe','Calendar_Year','Model_Year']

from pyspark.ml.feature import VectorAssembler
feat_assembler = VectorAssembler(inputCols = features_list, outputCol = 'features')
data = feat_assembler.transform(data)


from pyspark.ml.classification import LogisticRegression
import time
data_logi= data.select('features','not_zero','weight','Claim_Amount')
(trainingData, testData) = data_logi.randomSplit([0.7, 0.3], 47)
print('Data preprocessing finished.')
trainingData.cache()
testData.cache()

#classification
start = time.time()
print('Start training......')
logistic_Reg = LogisticRegression(labelCol ='not_zero',weightCol = 'weight',maxIter = 20)
logisticReg_model2 = logistic_Reg.fit(trainingData)

from pyspark.ml.evaluation import BinaryClassificationEvaluator

logisticReg_prediction = logisticReg_model2.transform(testData)

evaluator = BinaryClassificationEvaluator(labelCol="not_zero",metricName="areaUnderROC")
auc = evaluator.evaluate(logisticReg_prediction)
end = time.time()
print('Logistic Regression Execution time:',end-start)
print("auc = %g" % auc)

train_notzero = trainingData.filter('not_zero != 0')
test_notzero = testData.filter('not_zero != 0')

#training glm model
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol='Claim_Amount', maxIter=10, regParam=0.01,\
                                          family='Gamma', link='identity')
start = time.time()
glm_model = glm_poisson.fit(train_notzero)

#select zero sample
pred_zero = logisticReg_prediction.filter('prediction == 0')
pred_zero = pred_zero.withColumn('claim_prediction',pred_zero['not_zero']*0).select('Claim_Amount','claim_prediction')

#extract non zero value
pred_nonzero = logisticReg_prediction.filter('prediction != 0')
pred_nonzero = pred_nonzero.select('features','Claim_Amount')

#compare model with non zero value
pred_amount = glm_model.transform(pred_nonzero)
pred_amount = pred_amount.select('Claim_Amount','prediction')
pred_amount = pred_amount.withColumnRenamed('prediction','claim_prediction')

#union the zero and nonzero
result = pred_amount.union(pred_zero)
result = result.withColumn('Claim_Amount', result['Claim_Amount'].cast(DoubleType()))

#calculate score
evaluator = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="claim_prediction", metricName="rmse")
glm_rmse = evaluator.evaluate(result)
print('')
print("RMSE = %g" % glm_rmse)
print('')
end = time.time()
print('GLM Execution time:',end-start)
evaluator = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="claim_prediction", metricName="mae")
glm_mae = evaluator.evaluate(result)
print('')
print("MAE = %g" % glm_mae)
print('')
