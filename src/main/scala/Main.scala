import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.functions.{concat_ws, _}
import org.apache.spark.sql.SparkSession

object Main extends App {
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

  val mlModel = PipelineModel.read.load("v1_supervised")
  val results = mlModel.transform(df_flatten)
  //val results = df_json
    .select(
      col("parsed.CreatedAt").as("CreatedAt"),
      col("parsed.Id").as("Id"),
      //col("parsed.Text").as("Text"),
      col("parsed.FilteredText").as("sentence"),
      col("parsed.User.ScreenName").as("UserToken"),
      col("parsed.User.Location").as("UserLocation"),
      concat_ws("", col("class.result")).as("emotion")
      //concat_ws(",", col("parsed.HashtagEntities")).as("HashTags")
    )
  results.printSchema()

//  val writeStream = results.writeStream
//    .outputMode("update")
//    .format("console")
//    .option("truncate", "false")
//    .start()

  val writeStream = results
    .writeStream
    .format("csv")
    .option("checkpointLocation", sparkWriteCheckPoint)
    .option("path", "twitter.classified")
    //.option("sep", "\t")
    .start()
  writeStream.awaitTermination()
}