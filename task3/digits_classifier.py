import numpy as np
import random
import torch
from torch import nn, Tensor
from abc import ABC, abstractmethod
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier 


@dataclass
class ModelSettings:
    """Settings for model initialization"""
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_shape: tuple = (28, 28)
    n_estimators: int = 20


class DigitClassificationInterface(ABC):
    def __init__(self, settings: ModelSettings = None):
        self.settings = settings or ModelSettings()
        self._init_model()
    
    @abstractmethod
    def _init_model(self):
        """Initialize the model"""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> int:
        """
        Predicts the digit given an image.
        Args: image (np.ndarray): input image.
        Returns: int: predicted digit.
        """
        pass
    
    def train(self, data):
        """Base training method not implemented by default"""
        raise NotImplementedError("Training function is not implemented")
    

class CNNClassifier(DigitClassificationInterface):
    def _init_model(self):
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        ).to(self.settings.device)
    
    def predict(self, image: np.ndarray) -> int:
        if len(image.shape) == 2:
            image = image[None, None, :, :]  
        elif len(image.shape) == 3:
            image = image[None, :, :, :]
            
        with torch.no_grad():
            tensor_image = torch.FloatTensor(image).to(self.settings.device)
            output = self.model(tensor_image)
            prediction = output.argmax(dim=1).item()
        return prediction
    

class RandFClassifier(DigitClassificationInterface):
    def _init_model(self):
        # Generate dummy training data
        X_train = np.random.rand(100, 784) 
        y_train = np.random.randint(0, 10, size=100) 
        # Initialize and fit the model with dummy data
        self._model = RandomForestClassifier(n_estimators=self.settings.n_estimators)
        self._model.fit(X_train, y_train)
    
    def predict(self, image: np.ndarray) -> int:
        flattened = image.reshape(-1)
        return self._model.predict([flattened])[0]


class RandomClassifier(DigitClassificationInterface):
    def _init_model(self):
        pass 
    
    def predict(self, image: np.ndarray) -> int:
        return random.randint(0, 9)
    

class DigitClassifier:
    """Main classifier that wraps different implementations"""
    
    MODELS = {
        'cnn': CNNClassifier,
        'rf': RandFClassifier,
        'rand': RandomClassifier
    }
    
    def __init__(self, algorithm: str, settings: ModelSettings = None):
        """
        Initialize classifier with specified algorithm
        """
        if algorithm not in self.MODELS:
            raise ValueError(f"Algorithm must be one of {list(self.MODELS.keys())}")
            
        self.model = self.MODELS[algorithm](settings)
    
    def predict(self, image: np.ndarray) -> int:
        """
        Predict digit from image using selected algorithm
        """
        return self.model.predict(image)
    

if __name__ == '__main__':
   # Create random test image 
   image = np.random.rand(28, 28)
   
   # Test CNN classifier
   classifier = DigitClassifier(algorithm='cnn')
   prediction1 = classifier.predict(image)
   print(f"CNN prediction: {prediction1}")
   
   # Test Random Forest classifier
   classifier = DigitClassifier(algorithm='rf') 
   prediction2 = classifier.predict(image)
   print(f"Random Forest prediction: {prediction2}")

   # Test Random classifier
   classifier = DigitClassifier(algorithm='rand')
   prediction3 = classifier.predict(image)
   print(f"Random prediction: {prediction3}")
