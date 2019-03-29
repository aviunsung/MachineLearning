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
    val trainingDataPath="/home/avinash/personalRepo/MachineLearning/Throttling/train_dataset/Hour1_30thJan/*.tsv"
    val testDataPath="/home/avinash/personalRepo/MachineLearning/Throttling/test_dataset/155967.tsv"

    println("Loading Training Data....")
    var trainingData=loadData(trainingDataPath)
    trainingData=dataPreparation(trainingData)
    trainingData=weightedSampling(trainingData)

    println("Loading Test Data....")
    var testData=loadData(testDataPath)
    testData=dataPreparation(testData)

    //var Array(training,test)=trainingData.randomSplit(Array(0.6,0.4),seed=12345)

    println("Training Data Count::"+trainingData.count())
    println("Testing Data Count::"+testData.count())

    //trainingData=underSamplingMajorityClass(trainingData)
    println("Training Data Count After UnderSampling::"+trainingData.count())

    buildAndTestLRModel(trainingData, testData)
    //buildAndTestLRDecisionTreeModel(trainingData, testData)

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

  def weightedSampling(trainingData: DataFrame): DataFrame = {
    println("Performing Weighted Sampling....")
    var data=trainingData
    val numNegatives = data.filter(data("label") === 1).count
    val datasetSize = data.count
    val balancingRatio = (datasetSize - numNegatives).toDouble / datasetSize

    val calculateWeights = udf { d: Double =>
      if (d == 1) {
        1 * balancingRatio
      }
      else {
        (1 * (1.0 - balancingRatio))
      }
    }
    data = data.withColumn("classWeightCol", calculateWeights(data("label")))
    return data
  }

  def underSamplingMajorityClass(trainingData: DataFrame): DataFrame = {
    println("Performing Under Sampling....")
    var data = trainingData
    val majorityClassDF = data.filter("label=0")
    println("Majority Class (0) Count ::" + majorityClassDF.count)
    val minorityClassDF = data.filter("label=1")
    println("Minority Class (1) Count ::" + minorityClassDF.count)
    val sampleRatio = minorityClassDF.count().toDouble / data.count().toDouble
    println("Sampling Ratio::" + sampleRatio)
    val majorityClassSampledDF = majorityClassDF.sample(false, sampleRatio)
    println("Majority Class (0) Count After Under Sampling::" + majorityClassSampledDF.count)
    data = minorityClassDF.union(majorityClassSampledDF)
    return data
  }

  def dataPreparation(data: DataFrame):DataFrame = {
    //checkNullCount(data)
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
    //checkNullCount(cleanedData)
    return cleanedData
  }

  def featureTransformation(data: DataFrame): DataFrame = {
    println("Feature Transformation....")
    //Convert Categorical into Numerical values
    println("Label Encoding....")
    val tsIndexer=new StringIndexer().setInputCol("ts").setOutputCol("ts_index").fit(data)
    val adidIndexer=new StringIndexer().setInputCol("adid").setOutputCol("adid_index").fit(data)
    val gctryIndexer=new StringIndexer().setInputCol("gctry").setOutputCol("gctry_index").fit(data)
    val mkIndexer=new StringIndexer().setInputCol("mk").setOutputCol("mk_index").fit(data)
    val osIndexer=new StringIndexer().setInputCol("os").setOutputCol("os_index").fit(data)
    val loIndexer=new StringIndexer().setInputCol("lo").setOutputCol("lo_index").fit(data)
    //One-Hot Encoding
    println("One-Hot Encoding....")
    val encoder = new OneHotEncoderEstimator().setInputCols(Array("ts_index","adid_index","gctry_index","mk_index","os_index","lo_index"))
      .setOutputCols(Array("ts_vec","adid_vec","gctry_vec","mk_vec","os_vec","lo_vec"))

    //label-feature
    val featureCols =Array("ts_index","adid_index","adsize","wpfi","gctry_index","mk_index","os_index","lo_index","pp","pb")
    //val featureCols =Array("ts_vec","adid_vec","adsize","wpfi","gctry_vec","mk_vec","os_vec","lo_vec","pp","pb")
    val assembler=new VectorAssembler().setInputCols(featureCols).setOutputCol("features")

    val pipeline=new Pipeline().setStages(Array(tsIndexer,adidIndexer,gctryIndexer,mkIndexer,osIndexer,loIndexer,assembler))

    var transformedData=pipeline.fit(data).transform(data)
    println("Printing Transformed Data....")
    println(transformedData.show(5))
    return transformedData
  }

  def featureScaling(data: DataFrame): DataFrame = {
    //val scalerModel = new MinMaxScaler().setInputCol("features") .setOutputCol("scaledFeatures").fit(data)
    val scalerModel = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false).fit(data)
    //val scalerModel = new Normalizer().setInputCol("features").setOutputCol("scaledFeatures").setP(1.0)
    var scaledData=scalerModel.transform(data)
    println("Printing Scaled Data....")
    println(scaledData.show(5))
    return scaledData
  }

  def trainLRModel(trainingData: DataFrame):LogisticRegressionModel = {
    println("Training LR Model....")
    val lr=new LogisticRegression().setWeightCol("classWeightCol").setLabelCol("label").setFeaturesCol("scaledFeatures")
    val model=lr.fit(trainingData)
    return model
  }

  def testLRModel(model: LogisticRegressionModel, testData: DataFrame):DataFrame = {
    println("Testing LR Model...."+model)
    val predictions=model.transform(testData)
    return predictions
    //return printPredictions(predictions)
  }


  def trainDecisionTreeModel(trainingData: DataFrame) :DecisionTreeClassificationModel={
    println("Training Decision Tree Model....")
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("scaledFeatures")
    val model=dt.fit(trainingData)
    return model
  }

  def testDecisionTreeModel(model: DecisionTreeClassificationModel, testData: DataFrame):DataFrame = {
    println("Testing Decision Tree  Model...."+model)
    val predictions=model.transform(testData)
    return printPredictions(predictions)
  }

  private def printPredictions(predictions: DataFrame):DataFrame = {
    println("Printing Predicitons Data Schema....")
    predictions.printSchema()
    println("Prediction Count::" + predictions.groupBy("prediction").count.show)
    println("label Count" + predictions.groupBy("label").count.show)
    return predictions
  }

  def evaluateModel(predictions: DataFrame,modelName:String) = {
    println("Evaluating Model ::"+modelName)
    var predictionAndLabels= predictions.select(col("prediction"),col("label")).rdd.map{
      row => (row.getAs[Double](0), row.getAs[Integer](1).doubleValue())
    }
    val metrics=new MulticlassMetrics(predictionAndLabels)
    println("Confusion Matrix::")
    println(metrics.confusionMatrix)
    println("Recall of 1 :: "+metrics.recall(1.0))
    println("Recall of 0 :: "+metrics.recall(0.0))
  }
}
