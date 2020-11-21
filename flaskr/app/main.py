# import os
# import numpy as np
# import tensorflow as tf



# pip install pyspark
# from pyspark.sql import SparkSession
# spark = SparkSession \
#     .builder \
#     .getOrCreate()

    
# import pyspark

# Import dependencies
from flask import Flask, render_template, jsonify, request

from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
from pyspark.ml import Pipeline

from pyspark import SparkContext
sc = SparkContext("local")

# Create an instance of our Flask app.
app = Flask(__name__)


# Create all the features to the data set
pos_neg_to_num = StringIndexer(inputCol='action',outputCol='label')
tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
hashingTF = HashingTF(inputCol="stop_tokens", outputCol='hash_token')
idf = IDF(inputCol='hash_token', outputCol='idf_token')

# Create feature vectors
clean_up = VectorAssembler(inputCols=['idf_token', 'length'], outputCol='features')

# Create a and run a data processing Pipeline
data_prep_pipeline = Pipeline(stages=[pos_neg_to_num, tokenizer, stopremove, hashingTF, idf, clean_up])


# Set route
@app.route('/', methods=['post', 'get'])
def index():
    # Return the template with the teams list passed in
    headline=''
    if request.method=='POST':
        headline = request.form.get('headline')

        #Use headline to make a pysparkdf

        # Fit and transform the pipeline
        cleaner = data_prep_pipeline.fit(df)
        cleaned = cleaner.transform(df)

        # model.load('Trade_Predictor_model')
        # prediction = model.predict(cleaned)

        #transform 0, 1, 2 to buy/hold/sell

    return render_template('index.html', action = prediction)#, teams=teams)



if __name__ == "__main__":
    app.run(debug=True)