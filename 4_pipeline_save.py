import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

run_id = "0dc26514349d4d1192f35ba5d0a1cb44"
artifact_path = "pipeline_medical_insurance"
model_path = f"runs:/{run_id}/{artifact_path}"

pipeline = mlflow.sklearn.load_model(model_path)

output_dir = "pipeline_model"
mlflow.sklearn.save_model(sk_model=pipeline, path=output_dir)
print(f"Pipeline guardado en {output_dir}/")
