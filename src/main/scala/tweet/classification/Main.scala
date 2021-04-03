package tweet.classification

import org.apache.hadoop.conf.Configuration
import org.apache.spark.ml.PipelineModel
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object Main extends App {
  val kafkaHost = "localhost:9092"
  val sparkWriteCheckPoint = "write_checkpoint_tweet_classification"
  val inputKafkaTopic = "twitter.clean"
  val outputKafkaTopic = "twitter.classified"
  val spark = SparkSession.builder.appName("TweetClassification").master("local[*]").getOrCreate()
  val hadoopConfig: Configuration = spark.sparkContext.hadoopConfiguration
  hadoopConfig.set("fs.hdfs.impl", classOf[org.apache.hadoop.hdfs.DistributedFileSystem].getName)
  hadoopConfig.set("fs.file.impl", classOf[org.apache.hadoop.fs.LocalFileSystem].getName)
//  hadoopConfig.set("fs.s3n.impl", "org.apache.hadoop.fs.s3native.NativeS3FileSystem")
  // Read tweets from Kafka twitter.clean topic
  val readStream = spark
    .readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafkaHost)
    .option("subscribe", inputKafkaTopic)
    //    .option("startingOffsets", "earliest") // Always read from offset 0, for dev/testing purpose
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
      col("parsed.Id").cast("string").alias("key"), // key must be string or bytes
      to_json(struct(
        col("parsed.CreatedAt").as("CreatedAt"),
        col("parsed.Id").as("Id"),
        //col("parsed.Text").as("Text"),
        col("parsed.FilteredText").as("sentence"),
        col("parsed.User.ScreenName").as("UserToken"),
        col("parsed.User.Location").as("UserLocation"),
        concat_ws("", col("class.result")).as("emotion")
        //concat_ws(",", col("parsed.HashtagEntities")).as("HashTags")
      )).alias("value")
    )
  results.printSchema()

  // Write classified tweets to Kafka twitter.classified topic
  val writeStream = results
    .writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", kafkaHost)
    .option("topic", outputKafkaTopic)
    .option("checkpointLocation", sparkWriteCheckPoint)
    .start()
  writeStream.awaitTermination()
}
