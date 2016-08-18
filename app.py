import pandas as pd
import numpy as np
import sklearn as sklearn
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split as split_data
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Grab data from UCI Archive
data = pd.read_csv("./data.csv")
data.columns = [ "X", "Y", "Month", "Day", "FFMC", "DMC", "DC", "ISI", "Temp", "RH", "Wind", "Rain", "Area" ]


data = data.drop(["Temp", "RH", "Wind", "Rain"], axis=1)

# data = data[data["Area"] >= 30]
data = pd.get_dummies(data)
df = data

x = df.drop(["Area"], axis=1).as_matrix()
y = df["Area"].values

# pca = PCA(n_components=2)
# x = pca.fit_transform(x)
# print x

#correlogram

x_train, x_test, y_train, y_test = split_data(x, y, test_size=0.33, random_state=0)

# model = DecisionTreeRegressor(max_depth=5, max_features=None)
# model.fit(x_train, y_train)
#
# print
# print "Training accuracy:", model.score(x_train, y_train)
# print "Testing accuracy:", model.score(x_test, y_test)
# print

model = RandomForestRegressor(n_estimators=50, max_depth=10, max_features=None)
model.fit(x_train, y_train)

print "Training accuracy:", model.score(x_train, y_train)
print "Testing accuracy:", model.score(x_test, y_test)

"""
basic lecture about monitoring
how to monitor your own tools
puts links to gitswarm repos in confluence pages

willingness to learn
take suggestions well, coachable

respectful, thoughtful in words
emotional intelligence
humble, social and personal ability

should push the boundary
ask for forgiveness
don't wait for others
"""

# print "\n\n"
# print "Min area:", df["Area"].min(), "\n"
# print df.iloc[df["Area"].idxmin()], "\n\n"
#
# print "Max area:", df["Area"].max(), "\n"
# print df.iloc[df["Area"].idxmax()], "\n\n"
#
# sample = df.iloc[np.random.choice(df.index.values)]
# print "Sample area:", sample["Area"], "\n"
# print sample, "\n\n"
#
# print "Avg area:", df["Area"].mean(), "\n"
#
# fig, axarr = plt.subplots(2, 5)
#
# axarr[0, 0].scatter(df["FFMC"], df["Area"])
# axarr[0, 0].set_title("FFMC vs Area")
#
# axarr[0, 1].scatter(df["DMC"], df["Area"])
# axarr[0, 1].set_title("DMC vs Area")
#
# axarr[0, 2].scatter(df["DC"], df["Area"])
# axarr[0, 2].set_title("DC vs Area")
#
# axarr[0, 3].scatter(df["ISI"], df["Area"])
# axarr[0, 3].set_title("ISI vs Area")
#
# axarr[0, 4].scatter(df["X"], df["Y"])
# axarr[0, 4].set_title("Locations")
#
# axarr[1, 0].scatter(df["Temp"], df["Area"])
# axarr[1, 0].set_title("Temp vs Area")
#
# axarr[1, 1].scatter(df["RH"], df["Area"])
# axarr[1, 1].set_title("RH vs Area")
#
# axarr[1, 2].scatter(df["Wind"], df["Area"])
# axarr[1, 2].set_title("Wind vs Area")
#
# rpm = []
# months = data["Month"].unique()
# print months
# for month in data["Month"].unique():
# 	vals = []
# 	for area in data[data["Month"] == month]["Rain"]:
# 		vals.append(area)
# 	rpm.append(np.array(vals).mean())
#
# axarr[1, 3].bar(np.arange(len(rpm)), rpm)
# axarr[1, 3].set_title("Rain by Month")
# axarr[1, 3].set_xticks(np.arange(len(rpm)) + 0.2)
# axarr[1, 3].set_xticklabels(months, rotation=90)
#
# apm = []
# months = data["Month"].unique()
# print months
# for month in data["Month"].unique():
# 	vals = []
# 	for area in data[data["Month"] == month]["Area"]:
# 		vals.append(area)
# 	apm.append(np.array(vals).mean())
#
# axarr[1, 4].bar(np.arange(len(apm)), apm)
# axarr[1, 4].set_title("Area by Month")
# axarr[1, 4].set_xticks(np.arange(len(apm)) + 0.2)
# axarr[1, 4].set_xticklabels(months, rotation=90)
#
# plt.show()
