import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split as split_data

# Grab data from UCI Archive
df = pd.read_csv("./data/forest-fires.csv")
df.columns = [ "X", "Y", "Month", "Day", "FFMC", "DMC", "DC", "ISI", "Temp", "RH", "Wind", "Rain", "Area" ]
