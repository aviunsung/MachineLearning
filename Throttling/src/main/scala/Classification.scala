import org.apache.spark.ml.classification
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.classification.LogisticRegression

Logger.getLogger("org").setLevel(Level.ERROR)

val spark=SparkSession.builder().getOrCreate()
val data=spark.read.option("header","true").option("inferSchema","true").option("delimiter", "\t").format("csv").load("../../../train_dataset/Hour1_30thJan/155967.tsv")
data.printSchema()
data.columns
data.head(5)

for (row <- data.head(5)) {
  println(row)
}
println("First Row")
val firstRow=data.head(1)(0)
println("\n")