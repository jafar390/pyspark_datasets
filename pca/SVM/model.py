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
import sys



dataset = "/opt/dkube/dataset"
spark = SparkSession.builder.config("spark.driver.memory","15g").appName('imbalanced_multi_data').getOrCreate()
Logger= spark._jvm.org.apache.log4j.Logger
logger = Logger.getLogger(__name__)


if(len(sys.argv)< 3):
  logger.error("arguements are requires : note: run like this:python3 <filename.py> 0.6 0.4")
  sys.exit(1)

logger.error("########## train Split")
logger.error(sys.argv[1])
print(sys.argv[1])
logger.error("########## test Split")
logger.error(sys.argv[2])
print(sys.argv[2])

new_df = spark.read.csv(dataset + '/UNSW-Nb4.csv',header=True,inferSchema=True)


#Convert qualification and gender columns to numeric
col2_indexer = StringIndexer(inputCol="col2", outputCol="col2Index")
col4_indexer = StringIndexer(inputCol="col4",outputCol="col4Index")
col5_indexer = StringIndexer(inputCol="col5", outputCol="col5Index")
col6_indexer = StringIndexer(inputCol="col6", outputCol="col6Index")

#Convert qualificationIndex and genderIndex
logger.error("############ initiated onehot encoder algo")
onehot_encoder = OneHotEncoder(inputCols=["col2Index","col4Index","col5Index","col6Index"], outputCols=["col2_vec", "col4_vec","col5_vec","col6_vec"])
logger.error("############ onehot encoder done")
#Merge multiple columns into a vector column
logger.error("############ intiated vector assembler")
vector_assembler = VectorAssembler(inputCols=['col7','col8','col9','col10','col11','col12','col13','col15','col16','col17','col18','col19','col20','col21','col22','col23','col24','col25','col26','col27','col28','col29','col30','col31','col32','col33','col34','col35','col36','col37','col38','col39','col41','col42','col43','col44','col45','col46','col47',"col2_vec", "col4_vec","col5_vec","col6_vec"], outputCol='features')
logger.error("############ vector assembler done")
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

logger.error("#### before spliting")

train,test = df_transformed.randomSplit([float(sys.argv[1]), float(sys.argv[2])],seed=7)
#train60,test40 = df_transformed.randomSplit([0.6,0.4],seed=7)
#train70,test30 = df_transformed.randomSplit([0.7, 0.3], seed=7)
#train80,test20 = df_transformed.randomSplit([0.8,0.2],seed=7)
#train90,test10 = df_transformed.randomSplit([0.9,0.1],seed=7)

logger.error("#### after split")
logger.error("####  standardscaler on train dataset")

standardizer = StandardScaler(withMean=True, withStd=True,inputCol='features', outputCol='std_features')
standardizer_model = standardizer.fit(train)
standardized_features_df70 =standardizer_model.transform(train)

logger.error("#### standardscaler on test dataset")

standardizer = StandardScaler(withMean=True, withStd=True,inputCol='features', outputCol='std_features')
standardizer_model = standardizer.fit(test)
standardized_features_df30 =standardizer_model.transform(test)


from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA

logger.error("######  pca on standarded scaler using train")
pca = PCA(k=2, inputCol="std_features",outputCol="pca_features")
pca_model = pca.fit(standardized_features_df70)
pca_train = pca_model.transform(standardized_features_df70)

logger.error("###### pca on standarded scaler using test")

pca = PCA(k=2, inputCol="std_features",outputCol="pca_features")
pca_model = pca.fit(standardized_features_df30)
pca_test = pca_model.transform(standardized_features_df30)

logger.error("############# svm")

from pyspark.ml.classification import LinearSVC

# Define your classifier
lsvc = LinearSVC(maxIter=30, regParam=0.1,featuresCol="pca_features",labelCol="label")

stages209 = []
#stages += string_indexer
#stages += one_hot_encoder
#stages209 += [vector_assembler]
#stages209 += [minmax]
stages209 += [lsvc]

from pyspark.ml import Pipeline

pipeline209 = Pipeline().setStages(stages209)
svm7_model209 = pipeline209.fit(pca_train)

svm7_pp_df209 = svm7_model209.transform(pca_test)
svm7_predicited209 = svm7_pp_df209.select("prediction","label","rawPrediction")
tp = float(svm7_predicited209.filter("prediction == 1.0 AND label == 1").count())
fp = float(svm7_predicited209.filter("prediction == 1.0 AND label == 0").count())
tn = float(svm7_predicited209.filter("prediction == 0.0 AND label == 0").count())
fn = float(svm7_predicited209.filter("prediction == 0.0 AND label == 1").count())

acc = float((tp+tn)/svm7_predicited209.count())
pr = tp / (tp + fp)

re = tp / (tp + fn)

metrics209 = spark.createDataFrame([("TP",tp),("FP",fp),("TN",tn),("FN",fn),("accuracy",acc),("precision",pr),("Recall",re),("F1",2*pr*re/(re+pr))],["metric","value"])
metrics209.show()
