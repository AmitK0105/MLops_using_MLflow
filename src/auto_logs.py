import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

#set the tracking

mlflow.set_tracking_uri("http://127.0.0.1:5000")

#Load the dataset
wine= load_wine()
x= wine.data
y= wine.target

#train test split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.20, random_state=42)

#Define the params for RF model
max_depth=8
n_estimators=6

#set your experiment

mlflow.set_experiment("MLflow_experiments")
mlflow.autolog()

with mlflow.start_run():
    rf= RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(x_train, y_train)
    y_pred= rf.predict(x_test)
    accuracy_score1= accuracy_score(y_test, y_pred)

    # mlflow.log_metric("Accuracy",accuracy_score1)
    # mlflow.log_param("max_depth", max_depth)
    # mlflow.log_param("n_estimators", n_estimators)

    #creating a confusion matrxi
    cm= confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sn.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=wine.target_names,yticklabels=wine.target_names )
    plt.ylabel("Actual")
    plt.xlabel("predicted")
    plt.title("confusion_matrix")

    #save plot

    plt.savefig("confusion_matrix.png")

    # mlflow.log_artifact("confusion_matrix.png")
    mlflow.log_artifact(__file__)

    #set tags

    mlflow.set_tags({"Author" : "Amit", "Project": "Wine classification"})

    #log the model

    # mlflow.sklearn.log_model(rf, "RandomForest Model")

    print(accuracy_score1)