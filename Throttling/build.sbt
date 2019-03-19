name := "Throttling"

version := "0.1"

scalaVersion := "2.12.8"

logLevel:=Level.Error

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.12" % "2.4.0",
  "org.apache.spark" % "spark-sql_2.12" % "2.4.0",
  "org.apache.spark" % "spark-mllib_2.12" % "2.4.0"
)