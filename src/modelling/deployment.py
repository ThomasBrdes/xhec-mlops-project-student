from main import main
from prefect import serve

if __name__ == "__main__":
    train_model_workflow = main.to_deployment(
        name="train_model_flow",
        version="0.1.0",
        tags=["train"],
        interval=600,
    )

    try:
        serve(train_model_workflow)
    except Exception as e:
        print(f"Error occurred: {e}")
