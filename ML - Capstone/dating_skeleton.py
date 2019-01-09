import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from copy import deepcopy
import time
from sklearn.linear_model import LinearRegression

# df initialization
df = pd.read_csv("profiles.csv")
print(list(df))

# Income Mapping
income_mapping = {-1: 0, 20000: 0.02, 30000: 0.03, 40000: 0.04, 50000: 0.05, 60000: 0.06, 70000: 0.07, 80000: 0.08, 100000: 0.1, 150000: 0.15, 1000000: 1}
income_50k_mapping = {-1: 0, 20000: 0, 30000: 0, 40000: 0, 50000: 0, 60000: 1, 70000: 1, 80000: 1, 100000: 1, 150000: 1, 1000000: 1}
df["income_code"] = df.income.map(income_mapping)
df["income_50k"] = df.income.map(income_50k_mapping)


# Drug Mapping
drugs_mapping = {"never": 0, "sometimes": 0.5, "often": 1}
df["drugs_code"] = df.drugs.map(drugs_mapping)

# Drink Mapping
drink_mapping = {"not at all": 0, "rarely": 0.2, "socially": 0.4, "often": 0.6, "very often": 0.8, "desperately": 1}
df["drinks_code"] = df.drinks.map(drink_mapping)

# Status Mapping
status_mapping = {"single": 0, "available": 0, "seeing someone": 0.5, "married": 1, "unknown": 3}
df["status_code"] = df.status.map(status_mapping)
# Removing the unknowns
df = df[df['status_code'] < 2]

#Education Mapping
#If dropout, low 25%, working: 50%
# < High: 0, High: 0.2, College: 0.6 (2-year: 0.4), Masters: 0.8, Phd/Program: 1,
education_mapping = {"graduated from college/university": 0.6, "graduated from masters program": 0.8, "working on college/university": 0.4,
                     "working on masters program": 0.7, "graduated from two-year college": 0.4, "graduated from high school": 0.2,
                     "graduated from ph.d program": 1, "graduated from law school": 1, "working on two-year college": 0.3,
                     "dropped out of college/university": 0.3, "working on ph.d program": 0.9, "college/university": 0.6,
                     "graduated from space camp": 1, "dropped out of space camp": 0.85, "graduated from med school": 1,
                     "working on space camp": 0.9, "working on law school": 0.9, "two-year college": 0.4,
                     "working on med school": 0.9, "dropped out of two-year college": 0.25,"dropped out of masters program": 0.65,
                     "masters program": 0.8, "dropped out of ph.d program": 0.85, "dropped out of high school": 0.05,
                     "high school": 0.2, "working on high school": 0.1, "space camp": 1, "ph.d program": 1,
                     "law school": 1, "dropped out of law school": 0.85, "dropped out of med school": 0.85, "med school": 1}
df["education_code"] = df.education.map(education_mapping)

#Smoking Mapping
smokes_mapping = {"no":0, "sometimes":0.5, "when drinking": 0.3, "yes": 1, "trying to quit": 0.15}
df["smokes_code"] = df.smokes.map(smokes_mapping)

# Use in Regression
data = df[["income_code", "drugs_code", "status_code", "education_code", "age", "height"]]

#Use in Classification - is this new set of data below 50k in income?
#data = df[["income_50k", "drugs_code", "status_code", "education_code", "age", "height"]]



# Normalize!
x = data.values
minmax_scaler = preprocessing.MinMaxScaler()
x_normalized = minmax_scaler.fit_transform(x)

data = pd.DataFrame(x_normalized, columns=data.columns)
data = data.dropna()


#Change based on classification/regression
#labels = data[["income_50k"]]
labels = data[["income_code"]]


information = data[["drugs_code", "status_code", "education_code", "age", "height"]]
training_data, validation_data, training_labels, validation_labels = train_test_split(information, labels, random_state=1)


# Question: Is there a way to predict a user's income based on information?
pointer = []
pointer_values = []
training_data = training_data.values.reshape(-1,5)
validation_data = validation_data.values.reshape(-1,5)


training_labels = np.ravel(training_labels)
validation_labels = np.ravel(validation_labels)

x_range = range(5,45)
time1 = time.time()


kNR = KNeighborsRegressor(n_neighbors=34)
kNR.fit(training_data, training_labels)
pointer_values.append(deepcopy(kNR.score(validation_data, validation_labels)))

"""
# K-Means Classifier
for k in x_range:
    kNC = KNeighborsClassifier(n_neighbors=k)
    kNC.fit(training_data, training_labels)
    #print(kNC.score(validation_data, validation_labels))
    pointer_values.append(deepcopy(kNC.score(validation_data, validation_labels)))
"""
"""
# K-Means Regression
for k in x_range:
    kNR = KNeighborsRegressor(n_neighbors=k)
    kNR.fit(training_data, training_labels)
    #print(kNR.score(validation_data, validation_labels))
    pointer_values.append(deepcopy(kNR.score(validation_data, validation_labels)))

"""

"""
# Naive Bayes
gNB = GaussianNB()
gNB.fit(training_data, training_labels)
pointer = gNB.predict(validation_data)
#pointer_val.append(accuracy_score(validation_labels.round(), pointer.round()))
#pointer_val.append(recall_score(validation_labels.round(), pointer.round()))
pointer_values.append(accuracy_score(validation_labels.round(), pointer.round()))
"""

"""
# Linear Regression

lR = LinearRegression()
lR.fit(training_data, training_labels)
pointer_values.append(lR.score(validation_data, validation_labels))

print("Accuracy of model :: %s" % pointer_values[0])
"""

time2 = time.time()
print("Time taken: %s" % (time2 - time1))
print("Accuracy of model :: %s" % pointer_values[0])

"""
# Plotting Things
plt.plot(x_range, pointer_values)
plt.title("Accuracy of K-NeighborsClassifier")
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.show()
"""