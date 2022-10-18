import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# import our data as a dataframe
data = pd.read_csv("car.data")

# converts data into integers and also converts the pandas dataframe
# into multiple lists
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# what are we trying to predict/classify
predict = "class"

# creating our attributes and label variables
X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

# creating our train/test variables
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# creating an object for our model
model = KNeighborsClassifier(n_neighbors=7) # Parameter is the amount of neighbors we want to look for to classify

# training the model.
model.fit(X_train, y_train)

# test how accurate the model is
acc = model.score(X_test, y_test)

# display accuracy score
print(acc)

# Predict our test data
predicted = model.predict(X_test)

# Created a for loop to display what the model predicted, what our data was, and what the actual correct class was
# also used the names list to change the predicted class and the actual class into strings instead of numbers
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", X_test[x], "Actual: ", names[y_test[x]])

    # # print how far away each neighbor is from the data point and its index in its list
    # n = model.kneighbors([X_test[x]], 7, True)
    # print("N: ", n)