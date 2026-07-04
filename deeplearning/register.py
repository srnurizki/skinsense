import mlflow
import mlflow.keras
import tensorflow as tf
from config.settings import MLFLOW_TRACKING_URI, MLFLOW_DEFAULT_ARTIFACT_ROOT
from mlflow.exceptions import MlflowException

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
client = mlflow.tracking.MlflowClient()

try:
    experiment_id = mlflow.create_experiment(
        'sephora',
        artifact_location=MLFLOW_DEFAULT_ARTIFACT_ROOT)
except MlflowException:
    experiment_id = mlflow.get_experiment_by_name('sephora').experiment_id

mlflow.set_experiment(experiment_id=experiment_id)

# Skin Type
with mlflow.start_run():
    model = tf.keras.models.load_model('deeplearning/model_artifact/skin_type.keras')
    mlflow.keras.log_model(model, 'SkinTypeClassifier')
    run_id = mlflow.active_run().info.run_id

version_st = mlflow.register_model(f'runs:/{run_id}/SkinTypeClassifier', 'SkinTypeClassifier')
client.transition_model_version_stage('SkinTypeClassifier', version_st.version, 'Production')
print(f'SkinTypeClassifier v{version_st.version} -> Production')

# Skin Concern
with mlflow.start_run():
    model_sc = tf.keras.models.load_model('deeplearning/model_artifact/skin_concern.keras')
    mlflow.keras.log_model(model_sc, 'SkinConcernClassifier')
    run_id_sc = mlflow.active_run().info.run_id

version_sc = mlflow.register_model(f'runs:/{run_id_sc}/SkinConcernClassifier', 'SkinConcernClassifier')
client.transition_model_version_stage('SkinConcernClassifier', version_sc.version, 'Production')
print(f'SkinConcernClassifier v{version_sc.version} -> Production')