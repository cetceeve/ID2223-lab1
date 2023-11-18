import hopsworks
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib
import os
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold

project = hopsworks.login(project="zeihers_mart")
fs = project.get_feature_store()

# The feature view is the input set of features for your model. The features can come from different feature groups.    
# You can select features from different feature groups and join them together to create a feature view
wine_fg = fs.get_feature_group(name="wine", version=1)
query = wine_fg.select_except(features=["index"])
feature_view = fs.get_or_create_feature_view(name="wine",
                                  version=1,
                                  description="Read from Wine quality dataset",
                                  labels=["quality"],
                                  query=query)

# For testing only
# y = df["quality"] - 3 # negative 3 is because scikit-learn expects classes from 0
# X = df[["alcohol", "volatile acidity", "total sulfur dioxide", "chlorides", "density"]]
X, y = feature_view.training_data() 
print(X)
y = y - 3 # negative 3 is because scikit-learn expects classes from 0

# Train our model with SGBoost Decision Tree because we have a mix of continous and categorical data.
# It also shows good performance for outlier detection, which is what we need to find the good wines.
# With the early_stopping_rounds rounds parameter we can prevent overfitting 
model = xgb.XGBClassifier(tree_method="hist") 

# standard K-Fold cross validator but with defined seed
cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=132959)
# evaluate model
scores = cross_validate(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, return_train_score=True)


print('Training Accuracy: %.3f (%.3f)' % (scores["train_score"].mean(), scores["train_score"].std()))
print('Testing Accuracy: %.3f (%.3f)' % (scores["test_score"].mean(), scores["test_score"].std()))


# check which features are actually interesting
# xgb.plot_importance(model)

# We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
mr = project.get_model_registry()

# The contents of the 'wine_model' directory will be saved to the model registry. Create the dir, first.
model_dir="wine_model"
if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

# Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
joblib.dump(model, model_dir + "/wine_model.pkl")

# Specify the schema of the model's input/output using the features (X) and labels (y)
input_schema = Schema(X)
output_schema = Schema(y)
model_schema = ModelSchema(input_schema, output_schema)

# Create an entry in the model registry that includes the model's name, desc, metrics
wine_model = mr.python.create_model(
    name="wine_model", 
    metrics={"accuracy" : scores["test_score"].mean(), "standard deviation": scores["test_score"].std()},
    model_schema=model_schema,
    description="Wine quality Predictor"
)

# Upload the model to the model registry, including all files in 'model_dir'
wine_model.save(model_dir)