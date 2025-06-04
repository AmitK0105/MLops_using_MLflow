from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow

#Load the Breast Cancer datasets

data= load_breast_cancer()
x= pd.DataFrame(data.data, columns=data.feature_names)
y= pd.Series(data.target, name= "target")

#spliting into training & testing sets
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# Creating the Random Forest Classifier Model

rf= RandomForestClassifier(random_state=42)

# Defining the parameter grid  for GridsearchCV
param_grid= { "n_estimators": [10, 20, 30, 40 ,50],
             "max_depth":[5, 10, 12, 14, 15]}

#Applying GridSearchCV

gridsearch= GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

#Run without MLflow
# gridsearch.fit(x_train, y_train)

# #Displaing the best parameters and best scores

# best_param= gridsearch.best_params_
# best_score= gridsearch.best_score_

# print(best_param)
# print(best_score)

# By using MLflow

mlflow.set_experiment("Breast Cancer RF Model")

with mlflow.start_run():
    gridsearch.fit(x_train, y_train)

    #display the best parameter and best score

    best_param= gridsearch.best_params_
    best_score= gridsearch.best_score_

    #log parameters

    # mlflow.log_param(best_param, "Best_param_value")
    # log parameters
    for key, value in best_param.items():
        mlflow.log_param(key, value)

    # log metrics
    mlflow.log_metric("accuracy", best_score)

    # log training data
    train_df = x_train.copy()
    train_df["target"] = y_train
    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    # log test data
    test_df = x_test.copy()
    test_df["target"] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    # log source code
    mlflow.log_artifact(__file__)

    # log the best model
    mlflow.sklearn.log_model(gridsearch.best_estimator_, "random_forest")

    # set tags
    mlflow.set_tag("Author", "Amit Kumar")


    print(best_param)
    print(best_score)