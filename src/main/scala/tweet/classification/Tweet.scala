package tweet.classification

import org.apache.spark.sql.types.{ArrayType, LongType, StringType, StructType}

object Tweet {
  def schema(): StructType = {
    val userSchema = new StructType()
      .add("Id", LongType)
      .add("Name", StringType)
      .add("ScreenName", StringType) // Twitter User Handle
      .add("Location", StringType)
    val hashTagSchema = new StructType()
      .add("Text", StringType)

    new StructType()
      //.add("payload", new StructType()
      .add("CreatedAt", LongType)
      .add("Id", LongType)
      .add("Text", StringType)
      .add("User", userSchema)
      .add("HashtagEntities", ArrayType(hashTagSchema))
      .add("FilteredText", StringType)
    //      )
  }
}
