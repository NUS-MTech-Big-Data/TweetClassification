name := "TweetClassification"

version := "0.1"

scalaVersion := "2.11.12"

libraryDependencies ++= Seq(
  "com.johnsnowlabs.nlp" %% "spark-nlp" % "2.7.3",
  "org.apache.spark" %% "spark-sql" % "2.4.7",
  "org.apache.spark" %% "spark-mllib" % "2.4.7",
  "com.databricks" %% "spark-csv" % "1.2.0",
  "org.apache.spark" %% "spark-core" % "2.4.7",
  "org.apache.spark" %% "spark-sql" % "2.4.7",
  "org.apache.spark" %% "spark-sql-kafka-0-10" % "2.4.7"
)

lazy val commonSettings = Seq(
  version := "0.1-SNAPSHOT",
  organization := "nus.iss",
  scalaVersion := "2.11.12",
  test in assembly := {}
)
lazy val app = (project in file("app")).
  settings(commonSettings: _*).
  settings(
    mainClass in assembly := Some("tweet.classification.Main"),
    // more settings here ...
  )
assemblyMergeStrategy in assembly := {
  case "reference.conf" => MergeStrategy.concat
  case "META-INF/services/org.apache.spark.sql.sources.DataSourceRegister" => MergeStrategy.concat
  case PathList("META-INF", xs@_*) => MergeStrategy.discard
  case _ => MergeStrategy.first
}