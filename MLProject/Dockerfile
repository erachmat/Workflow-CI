FROM python:3.10-slim

WORKDIR /app

# Copy the saved model (already contains conda.yaml, python_env.yaml, requirements.txt,
# and the model.pkl -- generated automatically by mlflow.sklearn.save_model).
COPY saved_model /opt/ml/model

# Install exactly what MLflow recorded as this model's dependencies, plus mlflow itself
# to run the serving command. No conda, no apt-installed old Pythons -- just pip.
RUN pip install --no-cache-dir -r /opt/ml/model/requirements.txt \
    && pip install --no-cache-dir mlflow==2.19.0

EXPOSE 8080

# 'local' env-manager: dependencies are already installed in this image, so MLflow
# doesn't need to create (or fail to create) any additional environment at runtime.
CMD ["mlflow", "models", "serve", "-m", "/opt/ml/model", "-h", "0.0.0.0", "-p", "8080", "--env-manager", "local"]
