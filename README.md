# Stock Trade Predictor

![img](https://github.com/labulel/ML_StockTradePredictor/blob/main/flaskr/app/static/images/stock_market.jpg)<br>

## Analysts:

* Kamar Abdur-Rashid
* Preeti Vishwakarma
* Xinbei Luo


## Brief Background:

Technical Analysis is a trading discipline employed to evaluate investments and identify trading opportunities in price trends and patterns seen on charts. Technical analyst believe past trading activity and price changes of a security can be valuable indicators of the security's future price movements.
Financial news, market announcements, corporate news, or analyst forecasts form expectations and opinioons of investors and are reflected in volatile stock market reactions. However, current mainstream research and models do not factor in the role of media coverage (by way of disseminating information) in relation to financial performance. By using Machine Learning with Technical Analysis, we want to develop a model to determine the price pattern of stocks and suggest trade options.


## Data Sources:

1. NASDAQ Securities
2. Yahoo Finance Stock Data
3. Reuters Business News Headlines


## Extraction, Transform & Load:


### Extraction:

* Reuters Business News Headlines
  * Scrape Reuters website for headlines dating Jul 27, 2018 - Nov 16, 2020
* NASDAQ Securities
  * Downloadable CSV containing Securities information
  * Extract Symbol and Security Name
* Yahoo! Finance
  * API Request for Securities corresponding to Symbol News Headlines dates
  * Extract Open and Close Price


### Transformation:

* Reuters Business News Headlines
  * Advanced Natural Language Processing (NLP) using spaCy
  * Parse News Headlines with spaCy and return Organization Names
  * Join Headlines and Descriptions into Text Column
* NASDAQ Securities
  * Fuzzy String Matching using FuzzyWuzzy
  * Fuzzy match spaCy Organization Names against Security Names and return Symbol 
* Yahoo! Finance
  * Calculate Price Change and % Price Change from Open and Close Prices
  * Determine Trade Label (Buy, Sell, Hold) based on size of % Price Change
 
 
### Loading:

* Create an AWS S3 Bucket to host final Dataset in the cloud to train Machine Learning Model.


## Architecture:

* Used NaiveBayes Model for Trade prediction.
* Created Python Flask API to receive Trade Prediction when request is made.
* Used HTML/CSS to create from end interface.


## Conclusions:

* Accuracy of model at predicting trades is: 41.5456%
* 8.17% greater chance at predicting the correct trade vs. random selection
With Data 


## Program Challenges:

* Data Collection of News Headlines/ Summaries
* No datasets or APIs available
* Dirty Data
* Apache Spark installation and usage


## Program Insights:

* Train SpaCy NLP model with data to better predict organization names in headlines.
* Train the Trade Predictor model with more data to improve prediction accuracy.
* Obtain a PySpark Cluster to improve speed and efficiency of the Flask App

