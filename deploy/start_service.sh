#!/bin/zsh

service=$1

if [[ $service == "mlflow" ]]; then
    echo "Starting MLflow..."
    mlflow server \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root ./mlruns \
        --host 0.0.0.0 \
        --port 5001 \
        --workers 1 &
elif [[ $service == "airflow" ]]; then
    echo "Starting Airflow..."
    pkill -f airflow

    export PYTHONPATH="/Users/apple/Documents/PythonProject/collie:$PYTHONPATH"
    airflow db migrate

    airflow api-server --port 8080 &
    sleep 3
    airflow scheduler &
    sleep 3
    airflow triggerer &
    sleep 3
    airflow dag-processor &

fi