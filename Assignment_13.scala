#Code

class Calculator 
{
  def +(a: Int, b: Int): Int = a+b
  
  def -(a: Int, b: Int): Int = a-b
  
  def *(a: Int, b: Int): Long = a*b
  
  def /(a: Int, b: Int): Int = 
  {
    require(b != 0, "denominator can not be 0")    
    a/b
  }
}
 
object Calendar
{
  def main(args: Array[String]) =
  {
    val calc = new Calculator()
    
    println("Addition: " + calc.+(10, 2))
    println("Subtraction: " + calc.-(10, 2))
    println("Multiplication: " + calc.*(10, 2))
    println("Division: " + calc./(10, 2))
 
    //println("Division: " + calc./(10, 0))
  }
}




#Execution Steps
"""
## Open Terminal 1 and run following command
nc -lk 9999

## Open Terminal 2 and run following commands line by line

spark-shell

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import spark.implicits._

val spark = SparkSession.builder.appName("StructuredNetworkWordCount").getOrCreate()
  

val lines = spark.readStream.format("socket").option("host", "spark").option("port", 9999).load()
val words = lines.as[String].flatMap(_.split(" "))
val wordCounts = words.groupBy("value").count()

val query = wordCounts.writeStream.outputMode("complete").format("console").start()

## query.awaitTermination() //Most probably Not be needed

"""
