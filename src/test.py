import mlflow
print(mlflow.get_tracking_uri())



mlflow.set_tracking_uri("http://127.0.0.1:5000")
print(mlflow.get_tracking_uri())