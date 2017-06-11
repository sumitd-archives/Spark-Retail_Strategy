//package sales.data

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.sql._
import org.apache.log4j._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler, OneHotEncoder}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.regression.{LinearRegression}

    val xs = (0 until 100)
    val ys = (0 until 100).map { i => i + 5.0 * 4 }
    
    case class rows(UID:String, PID:String, Gender:String, Age:String, Occ:String, City:String,
    	Years:String, Marital:String, PC1:String, PC2:String, PC3:String, Purchase:Int)
    
    def mapper(line: String) : rows = {
    	val fields = line.split(',');
        val row:rows = rows(fields(0), fields(1), fields(2), fields(3), fields(4), fields(5), fields(6)
    	, fields(7), fields(8), fields(9), fields(10), fields(11).toInt)
          return row
    }    
     
    val modified_age = udf {(age : String) =>
        age match {
          case "0-17" => "1" 
          case "18-25" => "2"
          case "26-35" => "3"
          case "36-45" => "4"
          case "46-50" => "5"
          case "51-55" => "6"
          case "55+" => "7"
          case x => x
        }
     }
    
    val modified_years = udf {(years : String) =>
      if (years == "4+") "4" else years
    }
        
	
	Logger.getLogger("org").setLevel(Level.ERROR)
    var data = spark.read.format("CSV").option("header","true").load("/FileStore/tables/po7odqzf1490141921511/train.csv")
    data = data.withColumnRenamed("User_ID" , "uid")
               .withColumnRenamed("Product_ID", "pid")
               .withColumnRenamed("City_Category" , "city")
               .withColumnRenamed("Stay_In_Current_City_Years" , "years")
               .withColumnRenamed("Marital_Status" , "marital")
               .withColumnRenamed("Age" , "age")
               .withColumnRenamed("Product_Category_1", "pc1")
               .withColumnRenamed("Product_Category_2", "pc2")
               .withColumnRenamed("Product_Category_3", "pc3")
               .withColumnRenamed("Gender", "gender")
               .withColumnRenamed("Occupation", "occupation")
               .withColumnRenamed("Purchase", "label")
    
    data = data.withColumn("labeltemp", data("label").cast(DoubleType)).drop("label").withColumnRenamed("labeltemp", "label")
    //keep only those purchases > 5000 & < 20000
    //data = data.filter($"label" > 5000.0)
    //data = data.filter($"label" < 20000.0)
    data.printSchema()
    //display(data)
    
    val pid_indexer = new StringIndexer().setInputCol("pid").setOutputCol("pid_index")
    //val uid_indexer = new StringIndexer().setInputCol("uid").setOutputCol("pid_index")
    val city_indexer = new StringIndexer().setInputCol("city").setOutputCol("city_index")
    val years_indexer = new StringIndexer().setInputCol("years").setOutputCol("years_index")
    val marital_indexer = new StringIndexer().setInputCol("marital").setOutputCol("marital_index")
    val pc1_indexer = new StringIndexer().setInputCol("pc1").setOutputCol("pc1_index")
    val gender_indexer = new StringIndexer().setInputCol("gender").setOutputCol("gender_index")
    val occupation_indexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupation_index")
    val age_indexer = new StringIndexer().setInputCol("age").setOutputCol("age_index")
     
    val pid_encoder = new OneHotEncoder().setInputCol("pid_index").setOutputCol("pid_encoded")
    val city_encoder = new OneHotEncoder().setInputCol("city_index").setOutputCol("city_encoded")
    val years_encoder = new OneHotEncoder().setInputCol("years_index").setOutputCol("years_encoded")
    val marital_encoder = new OneHotEncoder().setInputCol("marital_index").setOutputCol("marital_encoded")
    val pc1_encoder = new OneHotEncoder().setInputCol("pc1_index").setOutputCol("pc1_encoded")
    val gender_encoder = new OneHotEncoder().setInputCol("gender_index").setOutputCol("gender_encoded")
    val occupation_encoder = new OneHotEncoder().setInputCol("occupation_index").setOutputCol("occupation_encoded")
    val age_encoder = new StringIndexer().setInputCol("age_index").setOutputCol("age_encoded")

    /*val assembler = new VectorAssembler()
              .setInputCols(Array("city_encoded", "years_encoded", "marital_encoded", "pc1_encoded", 
                                  "gender_encoded", "occupation_encoded"))
              .setOutputCol("features")*/

    val pipeline = new Pipeline().setStages(Array(pid_indexer, city_indexer, years_indexer, marital_indexer, pc1_indexer, gender_indexer, occupation_indexer,age_indexer, pid_encoder,city_encoder, years_encoder, marital_encoder, pc1_encoder, gender_encoder, occupation_encoder, age_encoder))//, assembler))
    val fitted = pipeline.fit(data)
    val transformed = fitted.transform(data)

    val dataframe = transformed.randomSplit(Array(0.6, 0.0, 0.2))
    val (training, validation, testing) = (dataframe(0), dataframe(1), dataframe(2))
    println(training.count)
    println(testing.count)
    println(validation.count)
    training.write.format("parquet").mode(SaveMode.Overwrite).save("/tmp/training.parquet")
    testing.write.format("parquet").mode(SaveMode.Overwrite).save("/tmp/testing.parquet")
    validation.write.format("parquet").mode(SaveMode.Overwrite).save("/tmp/validation.parquet")
    

