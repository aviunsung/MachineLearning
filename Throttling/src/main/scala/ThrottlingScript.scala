import org.apache.spark.ml.classification
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.sql.types.{IntegerType, StringType, StructField, StructType}
import org.apache.spark.sql.Row
import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql

Logger.getLogger("org").setLevel(Level.ERROR)

val spark=SparkSession.builder().config("spark.driver.cores",4).config("spark.driver.memory","4g").config("spark.executor.memory","4g").getOrCreate()

var data=spark.read.option("header","true").option("inferSchema","true").option("delimiter", "\t").format("csv").load("/home/avinash/personalRepo/MachineLearning/Throttling/train_dataset/Hour1_30thJan/156307.tsv")
data.printSchema()
data.columns.size
data.head(5)
data.count()
data.describe()

data=data.withColumnRenamed("verified_digit","label")
data.groupBy("label").count().show()


//Check Null Count Percentage
data.columns.foreach((col:String) => printNullCount(col))

def printNullCount(colName:String):Unit={
  var nullCount=data.filter(data(colName) === "NULL").count()
  println(colName+"::Null ="+nullCount)

}

/*********************DATA PRE-PROCESSING*****************************/
/********1. Handle Missing Values*********************************/
//drop complete null columns from dataframe

data=data.drop("id","pi","dab","ab","ai","ut","wcid","md")

//Feature selection
//drop repeating values columns
data=data.drop("pid","hour","sid","pfi","uh","je","los","oi","di","adtype")


import org.apache.spark.ml.feature._

//Convert Categorical into Numerical values
def execute_indexer(indexer:StringIndexer):sql.DataFrame={
  data = indexer.fit(data).transform(data)
  return data
}
val tsIndexer=new StringIndexer().setInputCol("ts").setOutputCol("ts_index").fit(data)
//data=execute_indexer(tsIndexer)
val adidIndexer=new StringIndexer().setInputCol("adid").setOutputCol("adid_index").fit(data)
//data=execute_indexer(adidIndexer)
val gctryIndexer=new StringIndexer().setInputCol("gctry").setOutputCol("gctry_index").fit(data)
//data=execute_indexer(gctryIndexer)
val mkIndexer=new StringIndexer().setInputCol("mk").setOutputCol("mk_index").fit(data)
//data=execute_indexer(mkIndexer)
val osIndexer=new StringIndexer().setInputCol("os").setOutputCol("os_index").fit(data)
//data=execute_indexer(osIndexer)
val loIndexer=new StringIndexer().setInputCol("lo").setOutputCol("lo_index").fit(data)
//data=execute_indexer(loIndexer)

//Feature Transformation label-feature
val featureCols =Array("ts_index","adid_index","adsize","wpfi","gctry_index","mk_index","os_index","lo_index","pp","pb")
val assembler=new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

val scalerModel = new MinMaxScaler().setInputCol("features") .setOutputCol("scaledFeatures").fit(data)
// rescale each feature to range [min, max].
//data = scalerModel.transform(data)

val Array(training,test)=data.randomSplit(Array(0.7,0.3),seed=12345)

import org.apache.spark.ml.Pipeline

val lr=new LogisticRegression().setLabelCol("label").setFeaturesCol("scaledFeatures")

val pipeline=new Pipeline().setStages(Array(tsIndexer,adidIndexer,gctryIndexer,mkIndexer,osIndexer,loIndexer,assembler,scalerModel,lr))

val model=pipeline.fit(training)

val predictions=model.transform(test)

predictions.printSchema()
predictions.groupBy("prediction").count.show
predictions.groupBy("label").count.show

//////Model Evaluation///////////////////
import org.apache.spark.mllib.evaluation.MulticlassMetrics
var predictionAndLabels= predictions.select($"prediction",$"label").as[(Double,Double)].rdd

val metrics=new MulticlassMetrics(predictionAndLabels)

println("Confusion Matrix:;")
println(metrics.confusionMatrix)