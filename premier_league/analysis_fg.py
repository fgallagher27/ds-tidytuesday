import pickle
import pandas as pd

pickle_in = open("data.pickle","rb")
long_df = pickle.load(pickle_in)

pd.set_option('display.max_columns', None)
print(long_df.head(10))

# may need log transformation
# either linear probability or logistic regression
# any kind of classificaton
# random forest - classification alogrith
# Naive bayes classification
# Neural network 

# feature importance procedure
# ridge regression