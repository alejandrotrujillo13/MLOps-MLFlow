import mlflow.sklearn

run_id = "401bc4ed7fb24f0a859850f10d887282"  # Reemplaza con tu RUN_ID
artifact_path = "pipeline_medical_insurance"
model_path = f"runs:/{run_id}/{artifact_path}"

loaded_model = mlflow.pyfunc.load_model(model_path)

output_dir = "pipeline_model"
mlflow.sklearn.save_model(sk_model=loaded_model._model_impl, path=output_dir)
