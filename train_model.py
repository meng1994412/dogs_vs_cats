# import packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import argparse
import pickle
import h5py

# construct a argument parser and parse the argument
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True, help = "path to HDF5 dataset")
ap.add_argument("-m", "--model", required = True, help = "path to output model")
ap.add_argument("-j", "--jobs", type = int, default = -1,
    help = "# of jobs to run when tuning hyperparameters")
args = vars(ap.parse_args())

# open the HDF5 database for reading then determine the index of the training and testing
# split, provided that this data was already shuffled prior to writing it to disk
db = h5py.File(args["db"], "r")
i = int(db["labels"].shape[0] * 0.75)

# define the set of parameters that we want to tune then start a
# grid search where we evaluate our model for each value of c
print("[INFO] tuning hyperparameters...")
params = {"C": [0.0001, 0.001, 0.01, 0.1, 1.0]}
model = GridSearchCV(LogisticRegression(), params, cv = 3, n_jobs = args["jobs"])
model.fit(db["features"][:i], db["labels"][:i])
print("[INFO] best hyperparameters: {}".format(model.best_params_))
# model = LogisticRegression(C = 0.001)
# model.fit(db["features"][:i], db["labels"][:i])
# print("[INFO] best hyperparameters: {}".format(0.001))

# generate a classification report for the model
print("[INFO] evaluating...")
preds = model.predict(db["features"][i:])
print(classification_report(db["labels"][i:], preds, target_names = db["label_names"]))

# compute the raw accuracy with extra precision
acc = accuracy_score(db["labels"][i:], preds)
print("[INFO] score: {}".format(acc))

print("[INFO] saving model...")
f = open(args["model"], "wb")
f.write(pickle.dumps(model.best_estimator_))
# f.write(pickle.dumps(model))
f.close()

# close the dataset
db.close()
