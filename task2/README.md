# Machine Learning Model for Target Prediction

Implementation of a machine learning solution for predicting a target variable from a tabular data. It includes EDA, data preprocessing, model training, and prediction using a Random Forest Regressor with RMSE as the evaluation metric.

## Install requirements
```bash
pip install -r requirements.txt
```

## EDA
`EDA.ipynb` contains exploratory data analysis. Data was checked for missing values, duplicates, feature distribution, correlation. 
Correlation merics showed us that column 8 correlates with column 6 a lot, so we checked the assumption if column 8 stores binary value that shows whether data in column 6 is positive or negative.
And it seemed to be true. So, due to high correlation we dropped column 8 to avoid multicollinearity in data.

## Train
For regression `Random Forest Regressor` was chosen. To get the best possible result randomized search for hyperparameters tuning was used along with k-fold cross-validation. 
In my case best parameters were `{'n_estimators': 400, 'min_samples_split': 5, 'min_samples_leaf': 2` which resulted in RMSE of `0.0036` on validation dataset.
Also, I tried `XGBoost` model to test both approaches: bagging and boosting, but the performance of XGBoost was quite poor. 

### Usage

To train model from terminal run `train.py` file and the best model will be saved in pickle format.

```python
python train.py <data_path> <output_model_path>
```
where `<data_path>` is path to the training dataset and `<output_model_path>` is path where the trained model will be serialized as .pkl.

## Predict
We load the trained model and make predictions on the test dataset. 

### Usage
To predict results from terminal run `predict.py` file

```python
python predict.py <data_path> <model_path> <output_path>
```
where `<data_path>` is path to the test dataset, `<model_path>` is path to the trained model pickle file, `<output_path>` is path where the predictions will be saved.

## Results
File `output_rf.csv` contains resulting prediction for unseen data from `hidden_test.csv'

