import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Matrix
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.types.DoubleType
import org.apache.log4j.Logger
import org.apache.log4j.Level
/**
  * Created by ALINA on 13.04.2018.
  */
object HousesPricesRegression {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val inputFile = args(0);

    //Initialize SparkSession
    val sparkSession = SparkSession
      .builder()
      .appName("spark-read-csv")
      .master("local[*]")
      .getOrCreate();

    //Read file to DF
    val pricesData = sparkSession.read
      .option("header", "true")
      .option("delimiter", ",")
      .option("nullValue", "")
      .option("treatEmptyValuesAsNulls", "true")
      .option("inferSchema", "true")
      .csv(inputFile)

    pricesData.show()

    val features = pricesData
      .select("MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "1stFlrSF", "2ndFlrSF",
        "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
        "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
        "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
        "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold") //, "SalePrice")

    features.createOrReplaceTempView("prices")

    val prices = pricesData.select("SalePrice").withColumn("SalePrice", pricesData("SalePrice").cast(DoubleType));
    val pricesRDD = prices.rdd.map(r => r.getDouble(0))

    val numFeatures = features.columns.length
    val corrType = "pearson"

    println(s"Correlation ($corrType) between label and each feature")
    println(s"Feature\tCorrelation")
    var feature = 0

    while (feature < numFeatures) {
      val column = features.columns(feature)
      val f = features.withColumn(column, features(column).cast(DoubleType))
      val featureRDD = f.select((column)).rdd.map(r => r.getDouble(0))
      val corr = Statistics.corr(featureRDD, pricesRDD, corrType)
      println(s"$column\t$corr")
      feature += 1
    }
  }
}
