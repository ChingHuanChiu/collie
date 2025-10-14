#!/bin/zsh

set -e
conda init zsh
source ~/.zshrc


# Define variables
CONDA_ENV="airflow"
AIRFLOW_HOME="$HOME/airflow"

# Activate the Conda environment
echo "🔧 Activating Conda environment: $CONDA_ENV..."
conda activate "$CONDA_ENV"

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install the latest MLflow
# echo "📦 Installing the latest MLflow..."
# pip install mlflow

# Install the latest Apache Airflow
echo "📦 Installing the latest Apache Airflow..."
pip install apache-airflow

# Set AIRFLOW_HOME
echo "📂 Setting AIRFLOW_HOME to $AIRFLOW_HOME"
export AIRFLOW_HOME=$AIRFLOW_HOME

# Initialize Airflow DB
echo "🧱 Initializing Airflow DB..."
airflow db migrate

# Activate FAB provider
pip install apache-airflow-providers-fab
export AIRFLOW__CORE__AUTH_MANAGER="airflow.providers.fab.auth_manager.fab_auth_manager.FabAuthManager"


# Create Airflow admin user
echo "👤 Creating Airflow admin user..."
airflow users create \
  --username admin \
  --password admin \
  --firstname ChingHuan \
  --lastname Chiu \
  --role Admin \
  --email stevenchiou8@gmail.com

# cat <<'EOF' > gunicorn.conf.py
# workers = 2
# timeout = 120
# graceful_timeout = 120
# loglevel = 'info'
# EOF



# Create Airflow user (admin / admin)
echo "👤 Creating Airflow user..."
airflow users create \
  --username admin \
  --password admin \
  --firstname ChingHuan \
  --lastname Chiu \
  --role Admin \
  --email stevenchiou8@gmail.com

# Display startup instructions
echo ""
echo "✅ Setup complete!"
echo ""
echo "👉 To start MLflow:"
echo "   conda activate $CONDA_ENV"
echo "   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"
echo ""
echo "👉 To start Airflow:"
echo "   conda activate airflow"
echo "   airflow api-server --port 8080 &   # open http://localhost:8080"
echo "   airflow scheduler &              # in a separate terminal"
