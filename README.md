1: Find some cool data.
2: Use Google Prediction API to build model.
3: Use API to get predictions from this model. 

I wanted to see if I we could predict the direction of an equity ( in this case looking at gold price through a gold ETF, specifically SPDR gold shares. ticker GLD)

Accessing the price data since 1-1-2010 until 4-30-2015 from yahoo Finance. Since we're probably not gonna be able to predict much just using the ticker data we need to add some features to this data. In get_and_clean_data.py we pull the price data and calculate some potentially useful indicators. We add this to a pandas DataFrame and then output data into correct format to test and train our model.


