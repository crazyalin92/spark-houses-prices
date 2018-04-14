import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.mllib.stat.Statistics
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.DoubleType
/**
  * Created by ALINA on 13.04.2018.
  */
object HousesPricesRegression {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val inputFile = "data/train.csv"

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


    var mssubclassIndex = new StringIndexer().setInputCol("MSSubClass").setOutputCol("MSSubClassIndex")
    var mszoningIndex = new StringIndexer().setInputCol("MSZoning").setOutputCol("MSZoningIndex")
    var lotfrontageIndex = new StringIndexer().setInputCol("LotFrontage").setOutputCol("LotFrontageIndex")
    var lotareaIndex = new StringIndexer().setInputCol("LotArea").setOutputCol("LotAreaIndex")
    var streetIndex = new StringIndexer().setInputCol("Street").setOutputCol("StreetIndex")
    var alleyIndex = new StringIndexer().setInputCol("Alley").setOutputCol("AlleyIndex")
    var lotshapeIndex = new StringIndexer().setInputCol("LotShape").setOutputCol("LotShapeIndex")
    var landcontourIndex = new StringIndexer().setInputCol("LandContour").setOutputCol("LandContourIndex")
    var utilitiesIndex = new StringIndexer().setInputCol("Utilities").setOutputCol("UtilitiesIndex")
    var lotconfigIndex = new StringIndexer().setInputCol("LotConfig").setOutputCol("LotConfigIndex")
    var landslopeIndex = new StringIndexer().setInputCol("LandSlope").setOutputCol("LandSlopeIndex")
    var neighborhoodIndex = new StringIndexer().setInputCol("Neighborhood").setOutputCol("NeighborhoodIndex")
    var condition1Index = new StringIndexer().setInputCol("Condition1").setOutputCol("Condition1Index")
    var condition2Index = new StringIndexer().setInputCol("Condition2").setOutputCol("Condition2Index")
    var bldgtypeIndex = new StringIndexer().setInputCol("BldgType").setOutputCol("BldgTypeIndex")
    var housestyleIndex = new StringIndexer().setInputCol("HouseStyle").setOutputCol("HouseStyleIndex")
    var overallqualIndex = new StringIndexer().setInputCol("OverallQual").setOutputCol("OverallQualIndex")
    var overallcondIndex = new StringIndexer().setInputCol("OverallCond").setOutputCol("OverallCondIndex")
    var yearbuiltIndex = new StringIndexer().setInputCol("YearBuilt").setOutputCol("YearBuiltIndex")
    var yearremodaddIndex = new StringIndexer().setInputCol("YearRemodAdd").setOutputCol("YearRemodAddIndex")
    var roofstyleIndex = new StringIndexer().setInputCol("RoofStyle").setOutputCol("RoofStyleIndex")
    var roofmatlIndex = new StringIndexer().setInputCol("RoofMatl").setOutputCol("RoofMatlIndex")
    var exterior1stIndex = new StringIndexer().setInputCol("Exterior1st").setOutputCol("Exterior1stIndex")
    var exterior2ndIndex = new StringIndexer().setInputCol("Exterior2nd").setOutputCol("Exterior2ndIndex")
    var masvnrtypeIndex = new StringIndexer().setInputCol("MasVnrType").setOutputCol("MasVnrTypeIndex")
    var masvnrareaIndex = new StringIndexer().setInputCol("MasVnrArea").setOutputCol("MasVnrAreaIndex")
    var exterqualIndex = new StringIndexer().setInputCol("ExterQual").setOutputCol("ExterQualIndex")
    var extercondIndex = new StringIndexer().setInputCol("ExterCond").setOutputCol("ExterCondIndex")
    var foundationIndex = new StringIndexer().setInputCol("Foundation").setOutputCol("FoundationIndex")
    var bsmtqualIndex = new StringIndexer().setInputCol("BsmtQual").setOutputCol("BsmtQualIndex")
    var bsmtcondIndex = new StringIndexer().setInputCol("BsmtCond").setOutputCol("BsmtCondIndex")
    var bsmtexposureIndex = new StringIndexer().setInputCol("BsmtExposure").setOutputCol("BsmtExposureIndex")
    var bsmtfintype1Index = new StringIndexer().setInputCol("BsmtFinType1").setOutputCol("BsmtFinType1Index")
    var bsmtfinsf1Index = new StringIndexer().setInputCol("BsmtFinSF1").setOutputCol("BsmtFinSF1Index")
    var bsmtfintype2Index = new StringIndexer().setInputCol("BsmtFinType2").setOutputCol("BsmtFinType2Index")
    var bsmtfinsf2Index = new StringIndexer().setInputCol("BsmtFinSF2").setOutputCol("BsmtFinSF2Index")
    var bsmtunfsfIndex = new StringIndexer().setInputCol("BsmtUnfSF").setOutputCol("BsmtUnfSFIndex")
    var totalbsmtsfIndex = new StringIndexer().setInputCol("TotalBsmtSF").setOutputCol("TotalBsmtSFIndex")
    var heatingIndex = new StringIndexer().setInputCol("Heating").setOutputCol("HeatingIndex")
    var heatingqcIndex = new StringIndexer().setInputCol("HeatingQC").setOutputCol("HeatingQCIndex")
    var centralairIndex = new StringIndexer().setInputCol("CentralAir").setOutputCol("CentralAirIndex")
    var electricalIndex = new StringIndexer().setInputCol("Electrical").setOutputCol("ElectricalIndex")
    var firstflrsfIndex = new StringIndexer().setInputCol("1stFlrSF").setOutputCol("1stFlrSFIndex")
    var secondflrsfIndex = new StringIndexer().setInputCol("2ndFlrSF").setOutputCol("2ndFlrSFIndex")
    var lowqualfinsfIndex = new StringIndexer().setInputCol("LowQualFinSF").setOutputCol("LowQualFinSFIndex")
    var grlivareaIndex = new StringIndexer().setInputCol("GrLivArea").setOutputCol("GrLivAreaIndex")
    var bsmtfullbathIndex = new StringIndexer().setInputCol("BsmtFullBath").setOutputCol("BsmtFullBathIndex")
    var bsmthalfbathIndex = new StringIndexer().setInputCol("BsmtHalfBath").setOutputCol("BsmtHalfBathIndex")
    var fullbathIndex = new StringIndexer().setInputCol("FullBath").setOutputCol("FullBathIndex")
    var halfbathIndex = new StringIndexer().setInputCol("HalfBath").setOutputCol("HalfBathIndex")
    var bedroomabvgrIndex = new StringIndexer().setInputCol("BedroomAbvGr").setOutputCol("BedroomAbvGrIndex")
    var kitchenabvgrIndex = new StringIndexer().setInputCol("KitchenAbvGr").setOutputCol("KitchenAbvGrIndex")
    var kitchenqualIndex = new StringIndexer().setInputCol("KitchenQual").setOutputCol("KitchenQualIndex")
    var totrmsabvgrdIndex = new StringIndexer().setInputCol("TotRmsAbvGrd").setOutputCol("TotRmsAbvGrdIndex")
    var functionalIndex = new StringIndexer().setInputCol("Functional").setOutputCol("FunctionalIndex")
    var fireplacesIndex = new StringIndexer().setInputCol("Fireplaces").setOutputCol("FireplacesIndex")
    var fireplacequIndex = new StringIndexer().setInputCol("FireplaceQu").setOutputCol("FireplaceQuIndex")
    var garagetypeIndex = new StringIndexer().setInputCol("GarageType").setOutputCol("GarageTypeIndex")
    var garageyrbltIndex = new StringIndexer().setInputCol("GarageYrBlt").setOutputCol("GarageYrBltIndex")
    var garagefinishIndex = new StringIndexer().setInputCol("GarageFinish").setOutputCol("GarageFinishIndex")
    var garagecarsIndex = new StringIndexer().setInputCol("GarageCars").setOutputCol("GarageCarsIndex")
    var garageareaIndex = new StringIndexer().setInputCol("GarageArea").setOutputCol("GarageAreaIndex")
    var garagequalIndex = new StringIndexer().setInputCol("GarageQual").setOutputCol("GarageQualIndex")
    var garagecondIndex = new StringIndexer().setInputCol("GarageCond").setOutputCol("GarageCondIndex")
    var paveddriveIndex = new StringIndexer().setInputCol("PavedDrive").setOutputCol("PavedDriveIndex")
    var wooddecksfIndex = new StringIndexer().setInputCol("WoodDeckSF").setOutputCol("WoodDeckSFIndex")
    var openporchsfIndex = new StringIndexer().setInputCol("OpenPorchSF").setOutputCol("OpenPorchSFIndex")
    var enclosedporchIndex = new StringIndexer().setInputCol("EnclosedPorch").setOutputCol("EnclosedPorchIndex")
    var threessnporchIndex = new StringIndexer().setInputCol("3SsnPorch").setOutputCol("3SsnPorchIndex")
    var screenporchIndex = new StringIndexer().setInputCol("ScreenPorch").setOutputCol("ScreenPorchIndex")
    var poolareaIndex = new StringIndexer().setInputCol("PoolArea").setOutputCol("PoolAreaIndex")
    var poolqcIndex = new StringIndexer().setInputCol("PoolQC").setOutputCol("PoolQCIndex")
    var fenceIndex = new StringIndexer().setInputCol("Fence").setOutputCol("FenceIndex")
    var miscfeatureIndex = new StringIndexer().setInputCol("MiscFeature").setOutputCol("MiscFeatureIndex")
    var miscvalIndex = new StringIndexer().setInputCol("MiscVal").setOutputCol("MiscValIndex")
    var mosoldIndex = new StringIndexer().setInputCol("MoSold").setOutputCol("MoSoldIndex")
    var yrsoldIndex = new StringIndexer().setInputCol("YrSold").setOutputCol("YrSoldIndex")
    var saletypeIndex = new StringIndexer().setInputCol("SaleType").setOutputCol("SaleTypeIndex")
    var saleconditionIndex = new StringIndexer().setInputCol("SaleCondition").setOutputCol("SaleConditionIndex")
    var salepriceIndex = new StringIndexer().setInputCol("SalePrice").setOutputCol("SalePriceIndex")


   /* val assembler = new VectorAssembler().setInputCols(Array(
      "MSSubClassIndex", "MSZoningIndex", "LotFrontageIndex", "LotAreaIndex", "StreetIndex",
      "AlleyIndex", "LotShapeIndex", "LandContourIndex", "UtilitiesIndex", "LotConfigIndex",
      "LandSlopeIndex", "NeighborhoodIndex", "Condition1Index", "Condition2Index", "BldgTypeIndex",
      "HouseStyleIndex", "OverallQualIndex", "OverallCondIndex", "YearBuiltIndex", "YearRemodAddIndex",
      "RoofStyleIndex", "RoofMatlIndex", "Exterior1stIndex", "Exterior2ndIndex", "MasVnrTypeIndex",
      "MasVnrAreaIndex", "ExterQualIndex", "ExterCondIndex", "FoundationIndex", "BsmtQualIndex",
      "BsmtCondIndex", "BsmtExposureIndex", "BsmtFinType1Index", "BsmtFinSF1Index", "BsmtFinType2Index",
      "BsmtFinSF2Index", "BsmtUnfSFIndex", "TotalBsmtSFIndex", "HeatingIndex", "HeatingQCIndex", "CentralAirIndex",
      "ElectricalIndex", "1stFlrSFIndex", "2ndFlrSFIndex", "LowQualFinSFIndex", "GrLivAreaIndex",
      "BsmtFullBathIndex", "BsmtHalfBathIndex", "FullBathIndex", "HalfBathIndex", "BedroomAbvGrIndex",
      "KitchenAbvGrIndex", "KitchenQualIndex", "TotRmsAbvGrdIndex", "FunctionalIndex", "FireplacesIndex",
      "FireplaceQuIndex", "GarageTypeIndex", "GarageYrBltIndex", "GarageFinishIndex", "GarageCarsIndex",
      "GarageAreaIndex", "GarageQualIndex", "GarageCondIndex", "PavedDriveIndex", "WoodDeckSFIndex",
      "OpenPorchSFIndex", "EnclosedPorchIndex", "3SsnPorchIndex", "ScreenPorchIndex", "PoolAreaIndex",
      "PoolQCIndex", "FenceIndex", "MiscFeatureIndex", "MiscValIndex", "MoSoldIndex", "YrSoldIndex",
      "SaleTypeIndex", "SaleConditionIndex", "SalePriceIndex"))
      .setOutputCol("indexedFeatures")*/

    val pipeline = new Pipeline()
      .setStages(Array(mssubclassIndex,	mszoningIndex,	lotfrontageIndex,	lotareaIndex,
        streetIndex,	alleyIndex,	lotshapeIndex,	landcontourIndex,	utilitiesIndex,
        lotconfigIndex,	landslopeIndex,	neighborhoodIndex,	condition1Index,	condition2Index,
        bldgtypeIndex,	housestyleIndex,	overallqualIndex,	overallcondIndex,	yearbuiltIndex,
        yearremodaddIndex,	roofstyleIndex,	roofmatlIndex,	exterior1stIndex,	exterior2ndIndex,
        masvnrtypeIndex,	masvnrareaIndex,	exterqualIndex,	extercondIndex,	foundationIndex,
        bsmtqualIndex,	bsmtcondIndex,	bsmtexposureIndex,	bsmtfintype1Index,	bsmtfinsf1Index,
        bsmtfintype2Index,	bsmtfinsf2Index,	bsmtunfsfIndex,	totalbsmtsfIndex,	heatingIndex,
        heatingqcIndex,	centralairIndex,	electricalIndex,	firstflrsfIndex,	secondflrsfIndex,
        lowqualfinsfIndex,	grlivareaIndex,	bsmtfullbathIndex,	bsmthalfbathIndex,	fullbathIndex,
        halfbathIndex,	bedroomabvgrIndex,	kitchenabvgrIndex,	kitchenqualIndex,	totrmsabvgrdIndex,
        functionalIndex,	fireplacesIndex,	fireplacequIndex,	garagetypeIndex,	garageyrbltIndex,
        garagefinishIndex,	garagecarsIndex,	garageareaIndex,	garagequalIndex,	garagecondIndex,
        paveddriveIndex,	wooddecksfIndex,	openporchsfIndex,	enclosedporchIndex,	threessnporchIndex,
        screenporchIndex,	poolareaIndex,	poolqcIndex,	fenceIndex,	miscfeatureIndex,	miscvalIndex,
        mosoldIndex,	yrsoldIndex,	saletypeIndex,	saleconditionIndex,	salepriceIndex))

    val model = pipeline.fit(pricesData)
    val newHouseData = model.transform(pricesData)

    //newHouseData.show(50)
    //newHouseData.printSchema()


    val features = newHouseData
      .select("MSSubClass", "LotArea", "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd",
        "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
        "1stFlrSF", "2ndFlrSF",
        "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
        "HalfBath", "BedroomAbvGr", "KitchenAbvGr",
        "TotRmsAbvGrd", "Fireplaces", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
        "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"

        , "MSZoningIndex", "LotFrontageIndex", "StreetIndex",
        "AlleyIndex", "LotShapeIndex", "LandContourIndex", "UtilitiesIndex", "LotConfigIndex",
        "LandSlopeIndex", "NeighborhoodIndex", "Condition1Index", "Condition2Index", "BldgTypeIndex",
        "HouseStyleIndex",
        "RoofStyleIndex", "RoofMatlIndex", "Exterior1stIndex", "Exterior2ndIndex", "MasVnrTypeIndex",
        "MasVnrAreaIndex", "ExterQualIndex", "ExterCondIndex", "FoundationIndex", "BsmtQualIndex",
        "BsmtCondIndex", "BsmtExposureIndex", "BsmtFinType1Index", "BsmtFinType2Index",
           "HeatingIndex", "HeatingQCIndex", "CentralAirIndex",
        "ElectricalIndex",

         "KitchenQualIndex",  "FunctionalIndex",
        "FireplaceQuIndex", "GarageTypeIndex", "GarageYrBltIndex", "GarageFinishIndex",
         "GarageQualIndex", "GarageCondIndex", "PavedDriveIndex",

        "PoolQCIndex", "FenceIndex", "MiscFeatureIndex",
        "SaleTypeIndex", "SaleConditionIndex") //, "SalePrice")

    features.createOrReplaceTempView("prices")

    val prices = pricesData.select("SalePrice").withColumn("SalePrice", pricesData("SalePrice").cast(DoubleType));
    val pricesRDD = prices.rdd.map(r => r.getDouble(0))

    val numFeatures = features.columns.length
    val corrType = "pearson"

    println(s"Correlation ($corrType) between label and each feature")
    println(s"Feature\tCorrelation")
    var feature = 0
    var k =List[(String,Double)]()
    while (feature < numFeatures) {
      val column = features.columns(feature)
      val f = features.withColumn(column, features(column).cast(DoubleType))
      val featureRDD = f.select((column)).rdd.map(r => r.getDouble(0))
      val corr = Statistics.corr(featureRDD, pricesRDD, corrType)
      k=k.::(column,corr)
      //println(s"$column\t$corr")
      feature += 1
    }
    k=k.sortBy(x=>Math.abs(x._2))
    k.foreach(println)
    val b=k.takeRight(17)
    println("Length: "+b.length)

    val s=b.map(x=>(x)._1)



    val assembler = new VectorAssembler().setInputCols(s.toArray)
      .setOutputCol("indexedFeatures")
    val vecdf = assembler.transform(newHouseData)
    vecdf.printSchema()
    //s.foreach(x=>println(x))
    val Array(training, test) = vecdf.randomSplit(Array(0.7, 0.3))

    val rf = new RandomForestRegressor()
      .setLabelCol("SalePrice")
      .setFeaturesCol("indexedFeatures")
      .setMaxDepth(10)
      .setMaxBins(100)

    val model2 = rf.fit(training)
    val predictions = model2.transform(test)

    predictions.select("prediction", "SalePrice", "indexedFeatures").show(5)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("SalePrice")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Root Mean Squared Error (RMSE) on test data = " + rmse)

  }
}
