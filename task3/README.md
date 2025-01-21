# Digits Classifier

Building `DigitClassifier` model to solve MNIST classification problem using different algorithms: `CNN`, `Random Forest`, `random value`.



## Installation

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

```python
from digit_classifier import DigitClassifier
import numpy as np

# Create random 28x28 image
image = np.random.rand(28, 28)

# Example CNN
classifier = DigitClassifier(algorithm='cnn')
prediction = classifier.predict(image)
print(f"CNN Prediction: {prediction}")

# Example Random Forest
classifier = DigitClassifier(algorithm='rf')
prediction = classifier.predict(image)
print(f"Random Forest Prediction: {prediction}")

# Example Random Classifier
classifier = DigitClassifier(algorithm='rand')
prediction = classifier.predict(image)
print(f"Random Prediction: {prediction}")
```
