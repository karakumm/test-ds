import sys
import pandas as pd
import pickle


def load_model(model_path):
    """
    Load the trained Random Forest model from a pickle file.
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def load_data(path):
    """
    Load and prepare test data for prediction.
    """
    data = pd.read_csv(path)
    data.drop(columns=['8'], inplace=True)
    return data


def predict(data_path, model_path, output_path):
    """
    Load the model and make predictions on unseen data
    """
    data = load_data(data_path)
    model = load_model(model_path)
    
    predictions = model.predict(data)
    
    output_df = pd.DataFrame({'Predictions': predictions})
    output_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 4, "Usage: python predict.py <data_path> <model_path> <output_path>"
    data_path = sys.argv[1]
    model_path = sys.argv[2]
    output_path = sys.argv[3]
    predict(data_path, model_path, output_path)
