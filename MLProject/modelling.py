import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn
import os
import shutil

def train_baseline():
    mlflow.autolog()

    print("Loading preprocessed data...")
    train_df = pd.read_csv('telco_churn_preprocessing/train.csv')
    test_df = pd.read_csv('telco_churn_preprocessing/test.csv')

    X_train = train_df.drop(columns=['Churn'])
    y_train = train_df['Churn']
    X_test = test_df.drop(columns=['Churn'])
    y_test = test_df['Churn']

    with mlflow.start_run(run_name="ci_cd_training"):
        print("Training Random Forest in MLflow Project...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        print(f"Test Accuracy: {score:.4f}")

        if os.path.exists("saved_model"):
            shutil.rmtree("saved_model")

        custom_env = {
            "name": "telco_env",
            "channels": ["conda-forge", "nodefaults"],
            "dependencies": [
                "python=3.10.12",
                "pip",
                {
                    "pip": [
                        "mlflow==2.19.0",
                        "scikit-learn==1.5.2",
                        "pandas==2.2.2",
                        "numpy==1.26.4"
                    ]
                }
            ]
        }

        mlflow.sklearn.save_model(model, "saved_model", conda_env=custom_env)
        print("Model saved locally to 'saved_model' directory.")

if __name__ == "__main__":
    train_baseline()
