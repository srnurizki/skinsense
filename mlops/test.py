# Import Libraries
import mlflow
from config.settings import MLFLOW_TRACKING_URI

# Set Experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment('sephora')