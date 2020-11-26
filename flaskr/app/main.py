# Import the dependencies
import pyspark
from flask import Flask, render_template, jsonify, request
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline
from pyspark.sql.functions import length
from pyspark import SparkContext
from pyspark.ml.classification import NaiveBayes, NaiveBayesModel

# pip install pyspark
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

# Create an instance of our Flask app.
app = Flask(__name__)

# Set route
@app.route('/', methods=['post', 'get'])
def index():
    # Return the template with the teams list passed in
    prediction=''
    if request.method=='POST':
        headline = request.form.get('headline')

        #Use headline to make a pysparkdf
        df = spark.createDataFrame([(0,headline)], ['ID','text'])
        
        # Create a length column to be used as a future feature 
        df = df.withColumn('length', length(df['text']))
        df.show()

        # Create all the features to the data set
        tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
        stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
        hashingTF = HashingTF(inputCol="stop_tokens", outputCol='hash_token')
        idf = IDF(inputCol='hash_token', outputCol='idf_token')

        # Create feature vectors
        clean_up = VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')

        # Create a and run a data processing Pipeline
        data_prep_pipeline = Pipeline(stages=[tokenizer, stopremove, hashingTF, idf, clean_up])
        
        # Fit and transform the pipeline
        cleaner = data_prep_pipeline.fit(df)
        cleaned = cleaner.transform(df)

        #load model and make prediction
        model = NaiveBayesModel.load('Trade_Predictor_Model')
        prediction = model.transform(cleaned).select('prediction').toPandas()
        prediction = prediction['prediction'].values[0]
        print(prediction)
        
        #transform 0, 1, 2 to hold/sell/buy
        if prediction == 0:
            prediction = 'Hold'
        elif prediction == 1:
            prediction = 'Sell'
        else:
            prediction = 'Buy'        

        print(prediction)

    return render_template('index.html', action = prediction)#, teams=teams)

if __name__ == "__main__":
    app.run(debug=True)