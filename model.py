import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score, train_test_split

from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor


""" PREPROCESS DATA FOR TRAINING """
def preprocess(df):
	np.random.seed(19)

	le_month = LabelEncoder()
	df["month"] = le_month.fit_transform(df["month"])

	le_day = LabelEncoder()
	df["day"] = le_day.fit_transform(df["day"])

	mms = MinMaxScaler()
	df["area"] = mms.fit_transform(df["area"].values.reshape(-1, 1))

	y = df.pop("area")
	X = df

	return X, y


""" DEFINE MODEL ARCHITECTURE """
def build_model():
    adam = Adam(lr=0.01)

    fire_in = Input((12,))
    dense = Dense(12, activation="sigmoid")(fire_in)
    dense = Dense(1,  activation="linear")(dense)

    model = Model(inputs=fire_in, outputs=dense)
    model.compile(optimizer=adam, loss="mse")
    return model


""" PERFORM CROSS VALIDATION """
def cross_validate(X, y):
	reg = KerasRegressor(build_fn=build_model, nb_epoch=20, verbose=0)
	return cross_val_score(reg, X.values, y.values, cv=10)


""" FIT MODEL TO DATA """
def train_model(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=19)

	model = build_model()
	model.fit(X_train, y_train, epochs=20, verbose=0, validation_data=(X_test, y_test))
	return model


""" MAIN ROUTINE """
def main():
	df = pd.read_csv("./data/forest-fires.csv")
	df.columns = [ "x_coord", "y_coord", "month", "day",
				   "ffmc", "dmc", "dc", "isi", "temp",
			  	   "rh", "wind", "rain", "area" ]

	X, y = preprocess(df)

	results = cross_validate(X, y)
	print("\nMean Cross Validation Score:", np.mean(results))

	model = train_model(X, y)
	model.save_weights("model_weights.h5")
	print("\nModel weights saved to: 'model_weights.h5'")


""" RUN PROGRAM """
if __name__ == "__main__":
	main()
