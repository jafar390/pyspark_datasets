from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler,StringIndexer,VectorIndexer,MinMaxScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder,CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
import logging
import sys

logger = logging.getLogger("py4j")

logger.info("############ this is test message")
logger.debug("############ this is test message")
logger.warning("############ this is test message")
logger.error("############ this is test message")

dataset = "/opt/dkube/dataset"
spark = SparkSession.builder.config("spark.driver.memory","15g").appName('imbalanced_multi_data').getOrCreate()
new_df = spark.read.csv(dataset + '/UNSW-Nb4.csv',header=True,inferSchema=True)


#Convert qualification and gender columns to numeric
col2_indexer = StringIndexer(inputCol="col2", outputCol="col2Index")
col4_indexer = StringIndexer(inputCol="col4",outputCol="col4Index")
col5_indexer = StringIndexer(inputCol="col5", outputCol="col5Index")
col6_indexer = StringIndexer(inputCol="col6", outputCol="col6Index")

#Convert qualificationIndex and genderIndex
onehot_encoder = OneHotEncoder(inputCols=["col2Index","col4Index","col5Index","col6Index"], outputCols=["col2_vec", "col4_vec","col5_vec","col6_vec"])

#Merge multiple columns into a vector column
vector_assembler = VectorAssembler(inputCols=['col7','col8','col9','col10','col11','col12','col13','col15','col16','col17','col18','col19','col20','col21','col22','col23','col24','col25','col26','col27','col28','col29','col30','col31','col32','col33','col34','col35','col36','col37','col38','col39','col41','col42','col43','col44','col45','col46','col47',"col2_vec", "col4_vec","col5_vec","col6_vec"], outputCol='features')
#Create pipeline and pass it to stages
pipeline = Pipeline(stages=[
           col2_indexer,col4_indexer, 
           col5_indexer,col6_indexer,
           onehot_encoder,
           vector_assembler
])
#fit and transform
df_transformed = pipeline.fit(new_df).transform(new_df)
df_transformed.show()


train60,test40 = df_transformed.randomSplit([0.6,0.4],seed=7)
train60,test40 = df_transformed.randomSplit([0.6,0.4],seed=7)
train70,test30 = df_transformed.randomSplit([0.7, 0.3], seed=7)
train80,test20 = df_transformed.randomSplit([0.8,0.2],seed=7)
train90,test10 = df_transformed.randomSplit([0.9,0.1],seed=7)



minmax = MinMaxScaler(inputCol="features",outputCol="normFeatures")
lr = LogisticRegression(labelCol="label",featuresCol="normFeatures",maxIter=10,regParam=0.3)

stages1 = []
stages1 +=[minmax]
stages1 += [lr]

logger.info("########train60,test40  logistic regression")

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages(stages1)
#model = pipeline.fit(train)
lr_model60 = pipeline.fit(train60)
lr_pp_df60 = lr_model60.transform(test40)

lr_predicited40 = lr_pp_df60.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(lr_predicited40.filter("prediction == 1.0 AND label == 1").count())
fp = float(lr_predicited40.filter("prediction == 1.0 AND label == 0").count())
tn = float(lr_predicited40.filter("prediction == 0.0 AND label == 0").count())
fn = float(lr_predicited40.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/lr_predicited40.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()


logger.info("####################LR with train70 and test30")

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages(stages1)
#model = pipeline.fit(train)
lr_model70 = pipeline.fit(train70)
lr_pp_df70 = lr_model70.transform(test30)

lr_predicited30 = lr_pp_df70.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(lr_predicited30.filter("prediction == 1.0 AND label == 1").count())
fp = float(lr_predicited30.filter("prediction == 1.0 AND label == 0").count())
tn = float(lr_predicited30.filter("prediction == 0.0 AND label == 0").count())
fn = float(lr_predicited30.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/lr_predicited30.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()


logger.info("$$$$$$$$$$LR with train80 and test20")

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages(stages1)
#model = pipeline.fit(train)
lr_model80 = pipeline.fit(train80)
lr_pp_df20 = lr_model80.transform(test20)

lr_predicited20 = lr_pp_df20.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(lr_predicited20.filter("prediction == 1.0 AND label == 1").count())
fp = float(lr_predicited20.filter("prediction == 1.0 AND label == 0").count())
tn = float(lr_predicited20.filter("prediction == 0.0 AND label == 0").count())
fn = float(lr_predicited20.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/lr_predicited20.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()


logger.info("##############LR with train90 and test10")

from pyspark.ml import Pipeline

pipeline = Pipeline().setStages(stages1)
#model = pipeline.fit(train)
lr_model90 = pipeline.fit(train90)
lr_pp_df90 = lr_model90.transform(test90)

lr_predicited10 = lr_pp_df90.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(lr_predicited10.filter("prediction == 1.0 AND label == 1").count())
fp = float(lr_predicited10.filter("prediction == 1.0 AND label == 0").count())
tn = float(lr_predicited10.filter("prediction == 0.0 AND label == 0").count())
fn = float(lr_predicited10.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/lr_predicited10.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()


logger.info("##################train60,test40 decision tree")

from pyspark.ml.classification import DecisionTreeClassifier
minmax = MinMaxScaler(inputCol="features",outputCol="normFeatures")
dt = DecisionTreeClassifier(featuresCol = 'normFeatures', labelCol = 'label', maxDepth = 3)

stages9 = []
#stages += string_indexer
#stages += one_hot_encoder
#stages9 += [vector_assembler]
stages9 += [minmax]
stages9 += [dt]

from pyspark.ml import Pipeline

pipeline19 = Pipeline().setStages(stages9)

dt_model60 = pipeline19.fit(train60)

dt_pp_df40 = dt_model60.transform(test40)

dt_predicited40 = dt_pp_df40.select("normFeatures","prediction","label","rawPrediction","probability")

tp = float(dt_predicited40.filter("prediction == 1.0 AND label == 1").count())
fp = float(dt_predicited40.filter("prediction == 1.0 AND label == 0").count())
tn = float(dt_predicited40.filter("prediction == 0.0 AND label == 0").count())
fn = float(dt_predicited40.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/dt_predicited40.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics1940 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics1940.show()

logger.info(" ############33decision tree train70 and test30")


from pyspark.ml import Pipeline

pipeline19 = Pipeline().setStages(stages9)

dt_model70 = pipeline19.fit(train70)

dt_pp_df30 = dt_model70.transform(test30)

dt_predicited30 = dt_pp_df30.select("normFeatures","prediction","label","rawPrediction","probability")

tp = float(dt_predicited30.filter("prediction == 1.0 AND label == 1").count())
fp = float(dt_predicited30.filter("prediction == 1.0 AND label == 0").count())
tn = float(dt_predicited30.filter("prediction == 0.0 AND label == 0").count())
fn = float(dt_predicited30.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/dt_predicited30.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics1940 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics1940.show()

logger.info("##################3Decision Tree Train80 and test20")


from pyspark.ml import Pipeline

pipeline19 = Pipeline().setStages(stages9)

dt_model80 = pipeline19.fit(train80)

dt_pp_df20 = dt_model80.transform(test20)

dt_predicited20 = dt_pp_df20.select("normFeatures","prediction","label","rawPrediction","probability")

tp = float(dt_predicited20.filter("prediction == 1.0 AND label == 1").count())
fp = float(dt_predicited20.filter("prediction == 1.0 AND label == 0").count())
tn = float(dt_predicited20.filter("prediction == 0.0 AND label == 0").count())
fn = float(dt_predicited20.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/dt_predicited20.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics1940 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics1940.show()

logger.info("###########Decision Tree train90 and test10")



from pyspark.ml import Pipeline

pipeline19 = Pipeline().setStages(stages9)

dt_model90 = pipeline19.fit(train90)

dt_pp_df10 = dt_model90.transform(test10)

dt_predicited10 = dt_pp_df10.select("normFeatures","prediction","label","rawPrediction","probability")

tp = float(dt_predicited10.filter("prediction == 1.0 AND label == 1").count())
fp = float(dt_predicited10.filter("prediction == 1.0 AND label == 0").count())
tn = float(dt_predicited10.filter("prediction == 0.0 AND label == 0").count())
fn = float(dt_predicited10.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/dt_predicited10.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics1940 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics1940.show()


logger.info("###############Random-Forest")

logger.info("Random-Forest train60 and test40")

from pyspark.ml.classification import RandomForestClassifier
minmax = MinMaxScaler(inputCol="features",outputCol="normFeatures")
rf = RandomForestClassifier(featuresCol = 'normFeatures', labelCol = 'label')

stages2 = []
#stages += string_indexer
#stages += one_hot_encoder
#stages2 += [vector_assembler]
stages2 += [minmax]
stages2 += [rf]

from pyspark.ml import Pipeline

pipeline2 = Pipeline().setStages(stages2)

rf_model60 = pipeline2.fit(train60)

rf_pp_df40 = rf_model60.transform(test40)

rf_predicited40 = rf_pp_df40.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(rf_predicited40.filter("prediction == 1.0 AND label == 1").count())
fp = float(rf_predicited40.filter("prediction == 1.0 AND label == 0").count())
tn = float(rf_predicited40.filter("prediction == 0.0 AND label == 0").count())
fn = float(rf_predicited40.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/rf_predicited40.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()

logger.info("################Random forest train70 and test30")

from pyspark.ml import Pipeline

pipeline2 = Pipeline().setStages(stages2)

rf_model70 = pipeline2.fit(train70)

rf_pp_df30 = rf_model70.transform(test30)

rf_predicited30 = rf_pp_df30.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(rf_predicited30.filter("prediction == 1.0 AND label == 1").count())
fp = float(rf_predicited30.filter("prediction == 1.0 AND label == 0").count())
tn = float(rf_predicited30.filter("prediction == 0.0 AND label == 0").count())
fn = float(rf_predicited30.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/rf_predicited30.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics30 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics30.show()

logger.info("###############Random forest train80 and test20")

from pyspark.ml import Pipeline

pipeline2 = Pipeline().setStages(stages2)

rf_model80 = pipeline2.fit(train80)

rf_pp_df20 = rf_model80.transform(test20)

rf_predicited20 = rf_pp_df20.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(rf_predicited20.filter("prediction == 1.0 AND label == 1").count())
fp = float(rf_predicited20.filter("prediction == 1.0 AND label == 0").count())
tn = float(rf_predicited20.filter("prediction == 0.0 AND label == 0").count())
fn = float(rf_predicited20.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/rf_predicited20.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics20 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics20.show()

logger.info("###############Random forest train90 and test10")

from pyspark.ml import Pipeline

pipeline2 = Pipeline().setStages(stages2)

rf_model90 = pipeline2.fit(train90)

rf_pp_df10 = rf_model90.transform(test10)

rf_predicited10 = rf_pp_df10.select("normFeatures","prediction","label","rawPrediction","probability")
tp = float(rf_predicited10.filter("prediction == 1.0 AND label == 1").count())
fp = float(rf_predicited10.filter("prediction == 1.0 AND label == 0").count())
tn = float(rf_predicited10.filter("prediction == 0.0 AND label == 0").count())
fn = float(rf_predicited10.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/rf_predicited10.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics10 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics10.show()



logger.info("#################3SVM with train60 and test40")

from pyspark.ml.classification import LinearSVC

# Define your classifier
minmax = MinMaxScaler(inputCol="features",outputCol="normFeatures")
lsvc = LinearSVC(maxIter=30, regParam=0.1,featuresCol="normFeatures",labelCol="label")

stages209 = []
#stages += string_indexer
#stages += one_hot_encoder
#stages209 += [vector_assembler]
stages209 += [minmax]
stages209 += [lsvc]


from pyspark.ml import Pipeline

pipeline209 = Pipeline().setStages(stages209)
svm_model60 = pipeline209.fit(train60)

svm_pp_df40 = svm_model60.transform(test40)

svm_predicited40 = svm_pp_df40.select("normFeatures","prediction","label","rawPrediction")
tp = float(svm_predicited40.filter("prediction == 1.0 AND label == 1").count())
fp = float(svm_predicited40.filter("prediction == 1.0 AND label == 0").count())
tn = float(svm_predicited40.filter("prediction == 0.0 AND label == 0").count())
fn = float(svm_predicited40.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/svm_predicited40.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()

logger.info("#########3svm train70 and test30")

pipeline209 = Pipeline().setStages(stages209)
svm_model70 = pipeline209.fit(train70)

svm_pp_df30 = svm_model70.transform(test30)

svm_predicited30 = svm_pp_df30.select("normFeatures","prediction","label","rawPrediction")
tp = float(svm_predicited30.filter("prediction == 1.0 AND label == 1").count())
fp = float(svm_predicited30.filter("prediction == 1.0 AND label == 0").count())
tn = float(svm_predicited30.filter("prediction == 0.0 AND label == 0").count())
fn = float(svm_predicited30.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/svm_predicited30.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics30 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics30.show()

logger.info("################3svm train80 and test20")

pipeline209 = Pipeline().setStages(stages209)
svm_model80 = pipeline209.fit(train80)

svm_pp_df20 = svm_model80.transform(test20)

svm_predicited20 = svm_pp_df20.select("normFeatures","prediction","label","rawPrediction")
tp = float(svm_predicited20.filter("prediction == 1.0 AND label == 1").count())
fp = float(svm_predicited20.filter("prediction == 1.0 AND label == 0").count())
tn = float(svm_predicited20.filter("prediction == 0.0 AND label == 0").count())
fn = float(svm_predicited20.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/svm_predicited20.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics20 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics20.show()

logger.info("######Svm train90 and test10")


pipeline209 = Pipeline().setStages(stages209)
svm_model90 = pipeline209.fit(train90)

svm_pp_df10 = svm_model90.transform(test10)

svm_predicited10 = svm_pp_df10.select("normFeatures","prediction","label","rawPrediction")
tp = float(svm_predicited10.filter("prediction == 1.0 AND label == 1").count())
fp = float(svm_predicited10.filter("prediction == 1.0 AND label == 0").count())
tn = float(svm_predicited10.filter("prediction == 0.0 AND label == 0").count())
fn = float(svm_predicited10.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/svm_predicited10.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics10 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics10.show()



logger.info("###########333GBN train60 and test40")

from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
minmax = MinMaxScaler(inputCol="features",outputCol="normFeatures")
gbt = GBTClassifier(labelCol="label", featuresCol="normFeatures", maxIter=10)
stages20912 = []
#stages += string_indexer
#stages += one_hot_encoder
#stages20912 += [vector_assembler]
stages20912 += [minmax]
stages20912 += [gbt]

from pyspark.ml import Pipeline

pipeline20912 = Pipeline().setStages(stages20912)
gbn_model60 = pipeline20912.fit(train60)

gbn_pp_df40 = gbn_model60.transform(test40)
gbn_predicited40 = gbn_pp_df40.select("normFeatures","prediction","label","rawPrediction")
tp = float(gbn_predicited40.filter("prediction == 1.0 AND label == 1").count())
fp = float(gbn_predicited40.filter("prediction == 1.0 AND label == 0").count())
tn = float(gbn_predicited40.filter("prediction == 0.0 AND label == 0").count())
fn = float(gbn_predicited40.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/gbn_predicited40.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics40 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics40.show()

logger.info("###########3gbn train70 and test30")

from pyspark.ml import Pipeline

pipeline20912 = Pipeline().setStages(stages20912)
gbn_model70 = pipeline20912.fit(train70)

gbn_pp_df30 = gbn_model70.transform(test30)
gbn_predicited30 = gbn_pp_df30.select("normFeatures","prediction","label","rawPrediction")
tp = float(gbn_predicited30.filter("prediction == 1.0 AND label == 1").count())
fp = float(gbn_predicited30.filter("prediction == 1.0 AND label == 0").count())
tn = float(gbn_predicited30.filter("prediction == 0.0 AND label == 0").count())
fn = float(gbn_predicited30.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/gbn_predicited30.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics30 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics30.show()

logger.info("###########3gbn train80 and test20")

from pyspark.ml import Pipeline

pipeline20912 = Pipeline().setStages(stages20912)
gbn_model80 = pipeline20912.fit(train80)

gbn_pp_df20 = gbn_model80.transform(test20)
gbn_predicited20 = gbn_pp_df20.select("normFeatures","prediction","label","rawPrediction")
tp = float(gbn_predicited20.filter("prediction == 1.0 AND label == 1").count())
fp = float(gbn_predicited20.filter("prediction == 1.0 AND label == 0").count())
tn = float(gbn_predicited20.filter("prediction == 0.0 AND label == 0").count())
fn = float(gbn_predicited20.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/gbn_predicited20.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics20 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics20.show()

logger.info("################gbn train90 and test10")



from pyspark.ml import Pipeline

pipeline20912 = Pipeline().setStages(stages20912)
gbn_model90 = pipeline20912.fit(train90)

gbn_pp_df10 = gbn_model90.transform(test10)
gbn_predicited10 = gbn_pp_df10.select("normFeatures","prediction","label","rawPrediction")
tp = float(gbn_predicited10.filter("prediction == 1.0 AND label == 1").count())
fp = float(gbn_predicited10.filter("prediction == 1.0 AND label == 0").count())
tn = float(gbn_predicited10.filter("prediction == 0.0 AND label == 0").count())
fn = float(gbn_predicited10.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/gbn_predicited10.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics10 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics10.show()




