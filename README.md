1. Find some cool data.
2. Use Google Prediction API to build model.
3. Use API to get predictions from this model. 

I wanted to see if I we could predict the direction of an equity ( in this case looking at gold price through a gold ETF, specifically SPDR gold shares. ticker GLD)

Accessing the price data since 1-1-2010 until 4-30-2015 from yahoo Finance. Since we're probably not going to be able to predict much just using the ticker data we need to come up with some additional features. 

In get_and_clean_data.py we pull the price data and calculate some potentially useful indicators. We add this to a pandas DataFrame and then output data into correct format. We then split up the data into testing data and training data and build the model with
the training data.

In main.py the script gets credentials from a json file. It reads data we want to test from previously saved testing data. Then it connects to the Prediction API using credentials and passes the features (but not the actual label) returing a prediction.

Since one prediction isn't particularly useful in this case there's an additional get_predictions_for_testing_data() function which gets a prediction for each row/day in testing data and logs which predictions were correct and which were not.  
