import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{LinearRegression}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.regression.DecisionTreeRegressionModel
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

var training = spark.read.format("parquet").option("header", "true").load("/tmp/training.parquet")
var testing = spark.read.format("parquet").option("header", "true").load("/tmp/testing.parquet")
val assembler = new VectorAssembler().setInputCols(Array("city_encoded", "years_encoded", "marital_encoded", "pc1_encoded", "gender_encoded", "occupation_encoded", "age_encoded" , "pid_encoded"))
.setOutputCol("indexed_features")
var training = assembler.transform(training)
var testing = assembler.transform(testing)

training = training.filter($"pc1" =!= "1" || $"label" > 5000)
training = training.filter($"pc1" =!= "5" || $"label" > 3000)
training = training.filter($"pc1" =!= "9" || $"label" > 7000)
training = training.filter($"pc1" =!= "10" || $"label" > 12000)
training = training.filter($"pc1" =!= "14" || $"label" > 6000)
training = training.filter($"pc1" =!= "16" || $"label" > 6000)

testing = testing.filter($"pc1" =!= "1" || $"label" > 5000)
testing = testing.filter($"pc1" =!= "5" || $"label" > 3000)
testing = testing.filter($"pc1" =!= "9" || $"label" > 7000)
testing = testing.filter($"pc1" =!= "10" || $"label" > 12000)
testing = testing.filter($"pc1" =!= "14" || $"label" > 6000)
testing = testing.filter($"pc1" =!= "16" || $"label" > 6000)

println(training.count)
training = training.select("pc1", "features" , "label")
testing = testing.select("pc1", "features" , "label")

var training_assembeled = featureIndexer.transform(training)
var testing_assembeled = featureIndexer.transform(testing)

var tr_pc_1 = training_assembeled.filter($"pc1" === "15" || $"pc1" === "7" || $"pc1" === "6" || $"pc1" === "9")
var tr_pc_2 = training_assembeled.filter($"pc1" === "10" )
var tr_pc_3 = training_assembeled.filter($"pc1" === "16" || $"pc1" === "1" || $"pc1" === "14")
var tr_pc_4 = training_assembeled.filter($"pc1" === "2" || $"pc1" === "3" || $"pc1" === "17")
var tr_pc_5 = training_assembeled.filter($"pc1" === "8" || $"pc1" === "11" || $"pc1" === "5")
var tr_pc_6 = training_assembeled.filter($"pc1" === "4" || $"pc1" === "18")
var tr_pc_7 = training_assembeled.filter($"pc1" === "20" || $"pc1" === "13" || $"pc1" === "12" || $"pc1" === "19")

var te_pc_1 = testing_assembeled.filter($"pc1" === "15" || $"pc1" === "7" || $"pc1" === "6" || $"pc1" === "9")
var te_pc_2 = testing_assembeled.filter($"pc1" === "10" )
var te_pc_3 = testing_assembeled.filter($"pc1" === "16" || $"pc1" === "1" || $"pc1" === "14")
var te_pc_4 = testing_assembeled.filter($"pc1" === "2" || $"pc1" === "3" || $"pc1" === "17")
var te_pc_5 = testing_assembeled.filter($"pc1" === "8" || $"pc1" === "11" || $"pc1" === "5")
var te_pc_6 = testing_assembeled.filter($"pc1" === "4" || $"pc1" === "18")
var te_pc_7 = testing_assembeled.filter($"pc1" === "20" || $"pc1" === "13" || $"pc1" === "12" || $"pc1" === "19")

var n_pc_1 = te_pc_1.count()
var n_pc_2 = te_pc_2.count()
var n_pc_3 = te_pc_3.count()
var n_pc_4 = te_pc_4.count()
var n_pc_5 = te_pc_5.count()
var n_pc_6 = te_pc_6.count()
var n_pc_7 = te_pc_7.count()

println(n_pc_1)
println(n_pc_2)
println(n_pc_3)
println(n_pc_4)
println(n_pc_5)
println(n_pc_6)
println(n_pc_7)


 val lr = new LinearRegression().setMaxIter(1000).setRegParam(0.01).setElasticNetParam(0.8).setFeaturesCol("indexedFeatures")

 val rf = new RandomForestRegressor()
  .setLabelCol("label")
  .setFeaturesCol("indexedFeatures")
  .setFeatureSubsetStrategy("0.4")
  .setNumTrees(50)
  .setMaxDepth(25)
  .setSubsamplingRate(0.5)
  .setCacheNodeIds(true)
  .setMinInstancesPerNode(10)
  .setMinInfoGain(0.001)
 // .setMaxBins(4000)*/

 
 val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("indexedFeatures")
  .setMaxDepth(20)
  .setMinInstancesPerNode(8)
  .setMinInfoGain(0.00001)

val model_1 = rf.fit(tr_pc_1) 
val model_2 = rf.fit(tr_pc_2)
val model_3 = rf.fit(tr_pc_3)
val model_4 = rf.fit(tr_pc_4)
val model_5 = rf.fit(tr_pc_5)
val model_6 = rf.fit(tr_pc_6)
val model_7 = rf.fit(tr_pc_7)

val test_predictions_1 = model_1.transform(te_pc_1)
val test_predictions_2 = model_2.transform(te_pc_2)
val test_predictions_3 = model_3.transform(te_pc_3)
val test_predictions_4 = model_4.transform(te_pc_4)
val test_predictions_5 = model_5.transform(te_pc_5)
val test_predictions_6 = model_6.transform(te_pc_6)
val test_predictions_7 = model_7.transform(te_pc_7)

val evaluator = new RegressionEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("mse")
  
val test_1_rmse = evaluator.evaluate(test_predictions_1)
val test_2_rmse = evaluator.evaluate(test_predictions_2)
val test_3_rmse = evaluator.evaluate(test_predictions_3)
val test_4_rmse = evaluator.evaluate(test_predictions_4)
val test_5_rmse = evaluator.evaluate(test_predictions_5)
val test_6_rmse = evaluator.evaluate(test_predictions_6)
val test_7_rmse = evaluator.evaluate(test_predictions_7)

println("rmse_1 = " + math.sqrt(test_1_rmse))
println("rmse_2 = " + math.sqrt(test_2_rmse))
println("rmse_3 = " + math.sqrt(test_3_rmse))
println("rmse_4 = " + math.sqrt(test_4_rmse))
println("rmse_5 = " + math.sqrt(test_5_rmse))
println("rmse_6 = " + math.sqrt(test_6_rmse))
println("rmse_7 = " + math.sqrt(test_7_rmse))

val se = test_1_rmse*n_pc_1 + test_2_rmse*n_pc_2 + test_3_rmse*n_pc_3 + test_4_rmse*n_pc_4 + test_5_rmse*n_pc_5 + test_6_rmse*n_pc_6 + test_7_rmse*n_pc_7

val mse = se / (n_pc_1 + n_pc_2 + n_pc_3 + n_pc_4 + n_pc_5 + n_pc_6 + n_pc_7)
val rmse = math.sqrt(mse)

println("RMSE test data = " + rmse)

