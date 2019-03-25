package model

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.classification._
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.{SparkConf, sql}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.Pipeline
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.ml.evaluation._


object Throttling {
  Logger.getLogger("org").setLevel(Level.ERROR)

  def main(args: Array[String]): Unit = {
    val trainingDataPath="/home/avinash/personalRepo/MachineLearning/Throttling/train_dataset/Hour1_30thJan/155967.tsv"
    val testDataPath="/home/avinash/personalRepo/MachineLearning/Throttling/test_dataset/155967.tsv"

    println("Loading Training Data....")
    var trainingData=loadData(trainingDataPath)
    println("Loading Test Data....")
    var testData=loadData(testDataPath)

    trainingData=dataPreparation(trainingData)
    testData=dataPreparation(testData)

    buildAndTestLRModel(trainingData, testData)
    buildAndTestLRDecisionTreeModel(trainingData,testData)

  }
  private def buildAndTestLRModel(trainingData: DataFrame, testData: DataFrame) = {
    var model = trainLRModel(trainingData)
    val predictions = testLRModel(model, testData)
    evaluateModel(predictions, "LR")
  }


  private def buildAndTestLRDecisionTreeModel(trainingData: DataFrame, testData: DataFrame) = {
    var model = trainDecisionTreeModel(trainingData)
    val predictions = testDecisionTreeModel(model, testData)
    evaluateModel(predictions, "Decision Tree")
  }

  def loadData(path:String): DataFrame = {
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName("Throttler")
    val spark = SparkSession.builder().config(conf).config("spark.driver.cores", 4).config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()
    var data = spark.read.option("header", "true").option("inferSchema", "true").option("delimiter", "\t").format("csv").load(path)
    println("Printing Data Schema....")
    data.printSchema()
    println("Total Columns::" + data.columns.size)
    println("Printing Top 5 Rows....")
    println(data.show(5))
    println("Total Rows::"+data.count())
    //println("Describing Data Schema....")
    //data.describe().show()
    //Rename Label Column for ML models
    data = data.withColumnRenamed("verified_digit", "label")
    println("Printing Label Count....")
    data.groupBy("label").count().show()
    return data
  }

  def dataPreparation(data: DataFrame):DataFrame = {
    checkNullCount(data)
    var processedData=handleMissingValues(data)
    processedData=featureTransformation(processedData)
    processedData=featureScaling(processedData)
    return processedData
  }

  private def checkNullCount(data: DataFrame) = {
    println("Checking Null Count....")
    data.columns.foreach((col: String) => printNullCount(col))
    def printNullCount(colName: String): Unit = {
      var nullCount = data.filter(data(colName) === "NULL").count()
      println(colName + "::Null Count=" + nullCount)

    }
  }

  def handleMissingValues(data:DataFrame):DataFrame = {
    println("Dropping complete Null columns::(\"id\", \"pi\", \"dab\", \"ab\", \"ai\", \"ut\", \"wcid\", \"md\")")
    //drop complete null columns from dataframe
    var cleanedData=data.drop("id", "pi", "dab", "ab", "ai", "ut", "wcid", "md")
    println("Dropping repeating values columns::\"pid\", \"hour\", \"sid\", \"pfi\", \"uh\", \"je\", \"los\", \"oi\", \"di\", \"adtype\"")
    //drop repeating values columns
    cleanedData=cleanedData.drop("pid", "hour", "sid", "pfi", "uh", "je", "los", "oi", "di", "adtype")
    checkNullCount(cleanedData)
    return cleanedData
  }

  def featureTransformation(data: DataFrame): DataFrame = {
    println("Feature Transformation....")
    //Convert Categorical into Numerical values
    def execute_indexer(data:DataFrame,indexer:StringIndexer):DataFrame={
      indexer.fit(data).transform(data)
      return data
    }
    println("Label Encoding....")
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

    val pipeline=new Pipeline().setStages(Array(tsIndexer,adidIndexer,gctryIndexer,mkIndexer,osIndexer,loIndexer,assembler))

    var transformedData=pipeline.fit(data).transform(data)
    println("Printing Transformed Data....")
    println(transformedData.show(5))
    return transformedData
  }

  def featureScaling(data: DataFrame): DataFrame = {
    val scalerModel = new MinMaxScaler().setInputCol("features") .setOutputCol("scaledFeatures").fit(data)
    //rescale each feature to range [min, max].
    var scaledData=scalerModel.transform(data)
    println("Printing Scaled Data....")
    println(scaledData.show(5))
    return scaledData
  }

  def trainLRModel(trainingData: DataFrame):LogisticRegressionModel = {
    println("Training LR Model....")
    val lr=new LogisticRegression().setLabelCol("label").setFeaturesCol("scaledFeatures")
    //val Array(training,test)=data.randomSplit(Array(0.7,0.3),seed=12345)
    val model=lr.fit(trainingData)
    return model
  }

  def trainDecisionTreeModel(trainingData: DataFrame) :DecisionTreeClassificationModel={
    println("Training Decision Tree Model....")
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label")
      .setFeaturesCol("scaledFeatures")
    val model=dt.fit(trainingData)
    return model
  }

  def testLRModel(model: LogisticRegressionModel, testData: DataFrame):DataFrame = {
    println("Testing LR Model....")
    val predictions=model.transform(testData)
    println("Printing Predicitons Data Schema....")
    predictions.printSchema()
    println("Prediction Count::"+predictions.groupBy("prediction").count.show)
    println("label Count"+predictions.groupBy("label").count.show)
    return predictions
  }

  def testDecisionTreeModel(model: DecisionTreeClassificationModel, testData: DataFrame):DataFrame = {
    println("Testing Decision Tree  Model....")
    val predictions=model.transform(testData)
    println("Printing Predicitons Data Schema....")
    predictions.printSchema()
    println("Prediction Count::"+predictions.groupBy("prediction").count.show)
    println("label Count"+predictions.groupBy("label").count.show)
    return predictions
  }


  def evaluateModel(predictions: DataFrame,modelName:String) = {
    println("Evaluating Model ::"+modelName)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println("Accuracy = " + accuracy)

    var predictionAndLabels= predictions.select(col("prediction"),col("label")).rdd.map{
      row => (row.getAs[Double](0), row.getAs[Integer](1).doubleValue())
    }
    val metrics=new MulticlassMetrics(predictionAndLabels)
    println("Confusion Matrix:;")
    println(metrics.confusionMatrix)
  }
}
