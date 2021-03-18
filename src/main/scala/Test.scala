import Main.df_flatten
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{concat_ws, _}

object Test extends App {
  val kafkaHost = "192.168.1.77:9092"
  val sparkWriteCheckPoint = "write_checkpoint"
  val spark = SparkSession.builder.appName("TweetClassification").master("local[*]").getOrCreate()
  // Read tweets from Kafka twitter.clean topic
  val readStream = spark
    .readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafkaHost)
    .option("subscribe", "twitter.clean")
    .option("startingOffsets", "earliest") // Always read from offset 0, for dev/testing purpose
    .option("failOnDataLoss", false)
    .load()
  readStream.printSchema()

  val df = readStream.selectExpr("CAST(value AS STRING)") // cast value from bytes to string
  val df_json = df.select(from_json(col("value"), Tweet.schema()).alias("parsed"))
  df_json.printSchema()

  val df_flatten = df_json
    .withColumn("sentence", col("parsed.FilteredText"))
    .filter(col("sentence").isNotNull)

  val mlModel = PipelineModel.read.load("v1_supervised_bert")
  val results = mlModel.transform(df_flatten)
    .select(
      col("sentence"),
      concat_ws("", col("class.result")).as("emotion")
    )

  // Write the output as a csv file
  val writeStream = results
    .writeStream
    .outputMode("append")
    .format("csv")
    .option("checkpointLocation", sparkWriteCheckPoint)
    .option("path", "tweets")
    .option("sep", "\t")
    .start()
  writeStream.awaitTermination()
}