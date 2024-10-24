def save_model(model, model_name):
    import pickle

    with open(f"{model_name}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved as {model_name}.pkl")
