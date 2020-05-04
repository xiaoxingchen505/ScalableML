from pyspark.sql import SparkSession
import numpy as np
spark = SparkSession.builder.master("local[2]")\
                            .appName("COM6012 Assignment Question——1")\
                            .config("spark.driver.memory", "50g")\
                            .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("OFF")

sc.setCheckpointDir('checkpoint/')

rawdata = spark.read.csv('Dataset/HIGGS.csv.gz')

feature_names = ['lepton pT', 'lepton eta', 'lepton phi', 
                 'missing energy magnitude', 'missing energy phi', 
                 'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 
                 'jet 2 pt', 'jet 2 eta', 'jet 2 phi', 'jet 2 b-tag', 
                 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 
                 'jet 4 pt', 'jet 4 eta', 'jet 4 phi', 'jet 4 b-tag', 
                 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

schemaNames = rawdata.schema.names

#Build up schemas
ncolumns = len(rawdata.columns)
rawdata = rawdata.withColumnRenamed(schemaNames[0],'labels')
for i in range(ncolumns-1):
     rawdata = rawdata.withColumnRenamed(schemaNames[i+1], feature_names[i])
schemaNames = rawdata.schema.names

#Change input data to double
from pyspark.sql.types import DoubleType
for i in range(ncolumns):
    rawdata = rawdata.withColumn(schemaNames[i], rawdata[schemaNames[i]].cast(DoubleType()))

#assemble input data to vector
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols = schemaNames[1:ncolumns], outputCol = 'features') 
raw_plus_vector = assembler.transform(rawdata)
data = raw_plus_vector.select('features','labels')

#random choose 5% of data for finding best parameter
small_data = data.sample(False, 0.05,47)
small_data.cache()

#define model pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

rf = RandomForestClassifier(featuresCol="features",labelCol="labels")

gbt = GBTClassifier(featuresCol="features",labelCol="labels")

from pyspark.ml import Pipeline

pipeline_rf = Pipeline(stages=[rf])
pipeline_gbt = Pipeline(stages=[gbt])


#setup training parameter grid

from pyspark.ml.tuning import ParamGridBuilder
import numpy as np


#parameter grid for random forest 
paramGrid_rf = ParamGridBuilder() \
    .addGrid(rf.numTrees, [3, 5, 7]) \
    .addGrid(rf.maxDepth, [3, 5, 7]) \
    .addGrid(rf.maxBins, [3, 5, 7])\
    .build()

#parameter grid for gradient boost 
paramGrid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [2,4,6]) \
    .addGrid(gbt.maxDepth, [3, 5, 7]) \
    .addGrid(gbt.maxBins, [3, 5, 7])\
    .build()

#training
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import time

evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

crossval_rf = CrossValidator(estimator=pipeline_rf,
                          estimatorParamMaps=paramGrid_rf,
                          evaluator=evaluator,
                          numFolds=3)

crossval_gbt = CrossValidator(estimator=pipeline_gbt,
                          estimatorParamMaps=paramGrid_gbt,
                          evaluator=evaluator,
                          numFolds=3)

(trainingData, testData) = small_data.randomSplit([0.8, 0.2],47)

start = time.time()
rfModel = crossval_rf.fit(trainingData)
rf_predictions = rfModel.transform(testData)
end = time.time()
print('Random Forest Finding best model execution time:',end-start)

start = time.time()
gbtModel = crossval_gbt.fit(trainingData)
gbt_predictions = gbtModel.transform(testData)
end = time.time()
print('GBT Finding best model execution time:',end-start)

#evaluate training score
evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

rf_accuracy_score = evaluator.evaluate(rf_predictions)

gbt_accuracy_score = evaluator.evaluate(gbt_predictions)

evaluator = BinaryClassificationEvaluator(labelCol="labels",rawPredictionCol='rawPrediction', metricName="areaUnderROC")

rf_auc_score = evaluator.evaluate(rf_predictions)

gbt_auc_score = evaluator.evaluate(gbt_predictions)

print('-----Random Forest Score-----')
print('accuracy score:',rf_accuracy_score)
print('AUC score:',rf_auc_score)
print('                             ')
print('--------GBT Score------------')
print('accuracy score:',gbt_accuracy_score)
print('AUC score:',gbt_auc_score)


bestPipeline_rf, bestPipeline_gbt = rfModel.bestModel, gbtModel.bestModel

bestModel_rf, bestModel_gbt = bestPipeline_rf.stages[0], bestPipeline_gbt.stages[0]

#output best parameters for each model.
print('-------Random Forest best model-----------')
print('numTrees - ', bestModel_rf.getNumTrees)
print('maxDepth - ', bestModel_rf.getOrDefault('maxDepth'))
print('maxBins - ', bestModel_rf.getOrDefault('maxBins'))
print('------------------------------------------')


print('---------------GBT best model-------------')
print('maxIter - ', bestModel_gbt.getOrDefault('maxIter'))
print('maxDepth - ', bestModel_gbt.getOrDefault('maxDepth'))
print('maxBins - ', bestModel_gbt.getOrDefault('maxBins'))
print('------------------------------------------')


#training with whole data with best parameter
print('----------------------------------')
print('Start to train with best parameter...')
print('----------------------------------')
best_rf = RandomForestClassifier(featuresCol="features",labelCol="labels",
                            numTrees = bestModel_rf.getNumTrees,
                           maxDepth = bestModel_rf.getOrDefault('maxDepth'),
                           maxBins = bestModel_rf.getOrDefault('maxBins'))

best_gbt = GBTClassifier(featuresCol="features",labelCol="labels",
                        maxIter=bestModel_gbt.getOrDefault('maxIter'),
                        maxDepth = bestModel_gbt.getOrDefault('maxDepth'),
                        maxBins = bestModel_gbt.getOrDefault('maxBins'))

(trainingData, testData) = data.randomSplit([0.8, 0.2],47)

trainingData.cache()
testData.cache()


start = time.time()
best_rfModel = best_rf.fit(trainingData)
best_rf_predictions = best_rfModel.transform(testData)
end = time.time()
print('Random Forest with best parameter execution time:',end-start)

start = time.time()
best_gbtModel = best_gbt.fit(trainingData)
best_gbt_predictions = best_gbtModel.transform(testData)
end = time.time()
print('GBT with best parameter execution time:',end-start)

#evalute training result
evaluator = MulticlassClassificationEvaluator(labelCol="labels", predictionCol="prediction", metricName="accuracy")

rf_accuracy_score = evaluator.evaluate(best_rf_predictions)

gbt_accuracy_score = evaluator.evaluate(best_gbt_predictions)

evaluator = BinaryClassificationEvaluator(labelCol="labels",rawPredictionCol='rawPrediction', metricName="areaUnderROC")

rf_auc_score = evaluator.evaluate(best_rf_predictions)

gbt_auc_score = evaluator.evaluate(best_gbt_predictions)

print('-----Random Forest Score-----')
print('accuracy score:',rf_accuracy_score)
print('AUC score:',rf_auc_score)
print('                             ')
print('--------GBT Score------------')
print('accuracy score:',gbt_accuracy_score)
print('AUC score:',gbt_auc_score)


#finding top 3 most relevant parameter
import pandas as pd
importances = best_rfModel.featureImportances
df_relevance = pd.DataFrame(importances.toArray())
df_relevance.columns = ['relevance']
df_relevance.index = feature_names
print('-------The most relevant Feature in random forest model---------')
print(df_relevance.sort_values(by=['relevance'], ascending=False).head(3))

importances = best_gbtModel.featureImportances
df_relevance = pd.DataFrame(importances.toArray())
df_relevance.columns = ['relevance']
df_relevance.index = feature_names
print('-------The most relevant Feature in GBT model--------------------')
print(df_relevance.sort_values(by=['relevance'], ascending=False).head(3))

