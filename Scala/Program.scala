object Operations {
       def largestNumber(a:Int,b:Int){
           if( a>b){
             println("Largest number is:" + a);
          }
          else{
              println("Largest number is:" + b);
          }
       }
       
       def PositiveNegative(a:Int){
            if( a>=0){
             println( a+ " is positive number");
          }
          else{
              println( a+ " is negative number");
          }
       }
       def EvenOdd(a:Int){
            if( a%2==0){
             println( a+ " is even number");
          }
          else{
              println( a+ " is odd number");
          }
       }
       
       
       def main(args: Array[String]) {
          var number1=20;
          var number2= -30;
          largestNumber(number1,number2);
          PositiveNegative(number1);
          PositiveNegative(number2);
          EvenOdd(20);
          EvenOdd(21);
       }
}


