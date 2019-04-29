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
  var weightMap = scala.collection.mutable.Map[String, scala.collection.mutable.Map[String,Double]]().withDefaultValue(null)


  def main(args: Array[String]): Unit = {
    val trainingDataPath="/home/avinash/personalRepo/MachineLearning/Throttling/train_dataset/Hour1_30thJan/156307.tsv"
    val testDataPath="/home/avinash/personalRepo/MachineLearning/Throttling/test_dataset/156307.tsv"

    println("Loading Training Data....")
    var trainingData=loadData(trainingDataPath)
    trainingData=dataPreparation(trainingData)
    //trainingData=weightedSampling(trainingData)

    //println("Loading Test Data....")
    //var testData=loadData(testDataPath)
   // testData=dataPreparation(testData)

    trainingData=underSamplingMajorityClass(trainingData)
    println("Training Data Count After UnderSampling::"+trainingData.count())

    //var training=trainingData
    //var test=testData
    var Array(training,test)=trainingData.randomSplit(Array(0.6,0.4),seed=12345)

    println("Training Data Count::"+training.count())
    println("Testing Data Count::"+test.count())
    buildAndTestLRModel(training, test)
    buildAndTestDecisionTreeModel(training, test)
    buildAndTestRandomForestModel(training, test)

  }

  private def buildAndTestLRModel(trainingData: DataFrame, testData: DataFrame) = {
    var model = trainLRModel(trainingData)
    val predictions = testLRModel(model, testData)
    evaluateModel(predictions, "LR")
  }


  private def buildAndTestDecisionTreeModel(trainingData: DataFrame, testData: DataFrame) = {
    var model = trainDecisionTreeModel(trainingData)
    val predictions = testDecisionTreeModel(model, testData)
    evaluateModel(predictions, "Decision Tree")
  }


  private def buildAndTestRandomForestModel(trainingData: DataFrame, testData: DataFrame) = {
    var model = trainRandomForestTreeModel(trainingData)
    val predictions = testRandomForestTreeModel(model, testData)
    evaluateModel(predictions, "Random Forest Tree")
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
    //data = data.withColumnRenamed("verified_digit", "label")
    data = data.withColumn("label", when(col("wcid")> 0, 1).otherwise(0))
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
    println("Dropping complete Null columns::(\"id\", \"pi\", \"dab\", \"ab\", \"ai\", \"ut\")")
    //drop complete null columns from dataframe
    var cleanedData=data.drop("id","pi","ut","ai","ab","dab")
    println("Dropping repeating values columns::\"ts\", \"hour\",\"pfi\", \"uh\", \"je\", \"los\", \"oi\", \"di\"")
    cleanedData=data.drop("ts","hour", "pfi", "uh", "je", "oi", "di","los")
    //checkNullCount(cleanedData)
    return cleanedData
  }

  def featureTransformation(data: DataFrame): DataFrame = {
    println("Feature Transformation....")
    var transformedData: DataFrame = performLabelEncoding(data)
    transformedData=assignFeatureWeightage(transformedData)
    transformedData=prepareFeatureVectors(transformedData)
    println("Printing Transformed Data....")
    println(transformedData.show(5))
    return transformedData
  }

  private def performLabelEncoding(data: DataFrame) = {
    val (pidIndexer,sidIndexer, adidIndexer,adsizeIndexer,adtypeIndexer, gctryIndexer, mkIndexer,mdIndexer, osIndexer, loIndexer) = performCategoricalLabelIndexing(data)
    val encoder = performOneHotEncoding(data)
    //val pipeline = new Pipeline().setStages(Array(pidIndexer,sidIndexer, adidIndexer,adsizeIndexer,adtypeIndexer, gctryIndexer, mkIndexer,mdIndexer, osIndexer, loIndexer))
    val pipeline = new Pipeline().setStages(Array(pidIndexer,loIndexer))
    var transformedData = pipeline.fit(data).transform(data)
    transformedData
  }

  def performCategoricalLabelIndexing(data: DataFrame) = {
    //Convert Categorical into Numerical values
    println("Label Encoding....")
    val pidIndexer=new StringIndexer().setInputCol("pid").setOutputCol("pid_index").setStringOrderType("alphabetAsc").fit(data)
    val sidIndexer=new StringIndexer().setInputCol("sid").setOutputCol("sid_index").setStringOrderType("alphabetAsc").fit(data)
    val adidIndexer=new StringIndexer().setInputCol("adid").setOutputCol("adid_index").setStringOrderType("alphabetAsc").fit(data)
    val adsizeIndexer=new StringIndexer().setInputCol("adsize").setOutputCol("adsize_index").setStringOrderType("alphabetAsc").fit(data)
    val adtypeIndexer=new StringIndexer().setInputCol("adtype").setOutputCol("adtype_index").setStringOrderType("alphabetAsc").fit(data)
    val gctryIndexer=new StringIndexer().setInputCol("gctry").setOutputCol("gctry_index").setStringOrderType("alphabetAsc").fit(data)
    val mkIndexer=new StringIndexer().setInputCol("mk").setOutputCol("mk_index").setStringOrderType("alphabetAsc").fit(data)
    val mdIndexer=new StringIndexer().setInputCol("md").setOutputCol("md_index").setStringOrderType("alphabetAsc").fit(data)
    val osIndexer=new StringIndexer().setInputCol("os").setOutputCol("os_index").setStringOrderType("alphabetAsc").fit(data)
    val loIndexer=new StringIndexer().setInputCol("lo").setOutputCol("lo_index")
    (pidIndexer,sidIndexer, adidIndexer,adsizeIndexer,adtypeIndexer, gctryIndexer, mkIndexer,mdIndexer, osIndexer, loIndexer)
  }


  def performOneHotEncoding(data: DataFrame): OneHotEncoderEstimator = {
    //One-Hot Encoding
    println("One-Hot Encoding....")
    val encoder = new OneHotEncoderEstimator().setInputCols(Array("ts_index", "adid_index", "gctry_index", "mk_index", "os_index", "lo_index"))
      .setOutputCols(Array("ts_vec", "adid_vec", "gctry_vec", "mk_vec", "os_vec", "lo_vec"))
    return encoder
  }

  def featureScaling(data: DataFrame): DataFrame = {
    val scalerModel = new MinMaxScaler().setInputCol("features") .setOutputCol("scaledFeatures").fit(data)
    //val scalerModel = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures").setWithStd(true).setWithMean(false).fit(data)
    //val scalerModel = new Normalizer().setInputCol("features").setOutputCol("scaledFeatures").setP(1.0)
    var scaledData=scalerModel.transform(data)
    println("Printing Scaled Data....")
    println(scaledData.show(5))
    return scaledData
  }

  def prepareFeatureVectors(data: DataFrame): DataFrame = {
    //val featureCols = Array( "ts_index","adid_index", "adsize", "adtype", "gctry_index", "mk_index","md_index", "os_index", "lo_index", "wpfi")
    //val featureCols = Array( "ts_index","adid_index", "adsize", "gctry_index", "mk_index","md_index", "os_index", "lo_index", "wpfi")

    val featureCols =Array("pid_index","sidWeightCol","adidWeightCol","adsizeWeightCol","adtypeWeightCol","gctryWeightCol","mkWeightCol","mdWeightCol","osWeightCol","lo_indexWeightCol","wpfi")
    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
    var transformedData = assembler.transform(data)
    return transformedData
  }

  def assignFeatureWeightage(transformedData: DataFrame): DataFrame = {
    var data=transformedData
    val datasetSize = data.count
    val totalOnes = data.filter(data("label") === 1).count
    val balancingRatio=(totalOnes.toDouble / datasetSize)

    println("datasetSize="+datasetSize)

    def calculateFeatureWeights(initialData: DataFrame,colName:String):scala.collection.mutable.Map[String, Double]={
      var weights=weightMap(colName)
      if(weights==null) {
        var data = initialData
        var counts = calculateFeatureCounts(data, colName)
        var res = data.filter(data("label") === 1)
        weights = scala.collection.mutable.Map[String, Double]().withDefaultValue(0)
        val rows = res.groupBy(colName).count().collect()
        rows.foreach { row =>
          var key = row.get(0).toString
          var totalCount = counts(key)
          var numOnes = row.get(1).toString.toDouble
          var numZeros = totalCount - numOnes
          //println("Key="+key+"  Total Count="+totalCount+"  numOnes="+numOnes+"   numZeroes="+numZeros)
          //var weight = (10 *(1.0 - balancingRatio)*(numOnes.toDouble) - 3 *balancingRatio*(numZeros.toDouble))
          var weight =(10*(numOnes.toDouble/totalCount)-3*(numZeros.toDouble/totalCount))*(totalOnes.toDouble/datasetSize)
          weights += (key -> weight)
          //println(key+"::"+weight)
        }
        weightMap += (colName -> weights)
      }
      if(colName=="gctry"){
        println("Column="+colName+"  weight="+weights("US"))
      }
      return weights
    }

    def calculateFeatureCounts(initialData: DataFrame,colName:String):scala.collection.mutable.Map[String, Double]={
      var data=initialData
      var counts = scala.collection.mutable.Map[String, Double]().withDefaultValue(0)
      val rows=data.groupBy(colName).count().collect()
      rows.foreach{ row =>
        var key =row.get(0).toString
        var count=row.get(1).toString.toDouble
        counts += (key -> count)
        //println(key+"::"+count)
      }
      return counts
    }

    def replaceWithFeatureWeights(initialData: DataFrame,weightsMap:scala.collection.mutable.Map[String, Double],colName:String):DataFrame={
      var data=initialData
      val assignFeatureWeights = udf { key: String =>
        weightsMap(key)
      }
      data = data.withColumn(colName+"WeightCol", assignFeatureWeights(data(colName)))
      //data.select(colName+"WeightCol").show
      return data
    }

    def assignFeatureWeights(initialData:DataFrame,colName:String):DataFrame={
      var data=initialData
      var featureWeights= calculateFeatureWeights(data,colName)
      data=replaceWithFeatureWeights(data,featureWeights,colName)
      return data
    }
    data=assignFeatureWeights(data,"sid")
    data=assignFeatureWeights(data,"adid")
    data=assignFeatureWeights(data,"adsize")
    data=assignFeatureWeights(data,"adtype")
    data=assignFeatureWeights(data,"gctry")
    data=assignFeatureWeights(data,"mk")
    data=assignFeatureWeights(data,"md")
    data=assignFeatureWeights(data,"os")
    data=assignFeatureWeights(data,"lo_index")
    return data
  }


  def trainLRModel(trainingData: DataFrame):LogisticRegressionModel = {
    println("Training LR Model....")
    val lr=new LogisticRegression()
      //.setWeightCol("classWeightCol")
      .setLabelCol("label").setFeaturesCol("scaledFeatures")
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
    val dt = new DecisionTreeClassifier().setLabelCol("label").setFeaturesCol("scaledFeatures").setThresholds(Array(1.1,1))
    val model=dt.fit(trainingData)
    return model
  }

  def testDecisionTreeModel(model: DecisionTreeClassificationModel, testData: DataFrame):DataFrame = {
    println("Testing Decision Tree  Model...."+model)
    val predictions=model.transform(testData)
    return predictions
    //return printPredictions(predictions)
  }

  def trainRandomForestTreeModel(trainingData: DataFrame) :RandomForestClassificationModel={
    println("Training Random Forest Model....")
    val dt = new RandomForestClassifier().setLabelCol("label").setFeaturesCol("scaledFeatures").setNumTrees(10)
    val model=dt.fit(trainingData)
    return model
  }

  def testRandomForestTreeModel(model: RandomForestClassificationModel, testData: DataFrame):DataFrame = {
    println("Testing Decision Tree  Model...."+model)
    val predictions=model.transform(testData)
    return predictions
    //return printPredictions(predictions)
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