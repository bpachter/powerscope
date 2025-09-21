"""
PyTorch neural network models for electricity demand forecasting.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


class ElectricityDemandDataset(Dataset):
    """PyTorch dataset for electricity demand data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Features array
            y: Target array
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class QuantileLoss(nn.Module):
    """Quantile loss function."""
    
    def __init__(self, quantile: float):
        """
        Initialize quantile loss.
        
        Args:
            quantile: Quantile level (0-1)
        """
        super().__init__()
        self.quantile = quantile
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute quantile loss.
        
        Args:
            pred: Predictions
            target: True values
            
        Returns:
            Quantile loss
        """
        error = target - pred
        loss = torch.where(error >= 0, 
                          self.quantile * error,
                          (self.quantile - 1) * error)
        return loss.mean()


class DeepQuantileRegressor(nn.Module):
    """Deep neural network for quantile regression."""
    
    def __init__(self, 
                 input_size: int,
                 hidden_sizes: List[int] = [128, 64, 32],
                 dropout: float = 0.2,
                 num_quantiles: int = 3):
        """
        Initialize deep quantile regressor.
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout probability
            num_quantiles: Number of quantiles to predict
        """
        super().__init__()
        
        self.input_size = input_size
        self.num_quantiles = num_quantiles
        
        # Build layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
            
        # Output layer for each quantile
        layers.append(nn.Linear(prev_size, num_quantiles))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features
            
        Returns:
            Quantile predictions
        """
        return self.network(x)


class LSTMQuantileRegressor(nn.Module):
    """LSTM neural network for time series quantile regression."""
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.2,
                 num_quantiles: int = 3):
        """
        Initialize LSTM quantile regressor.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_quantiles: Number of quantiles to predict
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_quantiles = num_quantiles
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_size, num_quantiles)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input sequences [batch_size, seq_len, input_size]
            
        Returns:
            Quantile predictions [batch_size, num_quantiles]
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and output layer
        output = self.dropout(last_output)
        output = self.output_layer(output)
        
        return output


class TorchQuantileTrainer:
    """Trainer for PyTorch quantile models."""
    
    def __init__(self,
                 model: nn.Module,
                 quantiles: List[float] = [0.1, 0.5, 0.9],
                 learning_rate: float = 0.001,
                 device: str = 'cpu'):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            quantiles: List of quantiles to predict
            learning_rate: Learning rate
            device: Device to use ('cpu' or 'cuda')
        """
        self.model = model.to(device)
        self.quantiles = quantiles
        self.device = device
        
        # Create loss functions for each quantile
        self.loss_functions = {
            i: QuantileLoss(q) for i, q in enumerate(quantiles)
        }
        
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch_x)
            
            # Calculate loss for each quantile
            batch_loss = 0.0
            for i, loss_fn in self.loss_functions.items():
                pred_quantile = predictions[:, i]
                batch_loss += loss_fn(pred_quantile, batch_y)
                
            # Backward pass
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            
        return total_loss / len(dataloader)
        
    def validate_epoch(self, dataloader: DataLoader) -> float:
        """
        Validate for one epoch.
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                
                # Calculate loss for each quantile
                batch_loss = 0.0
                for i, loss_fn in self.loss_functions.items():
                    pred_quantile = predictions[:, i]
                    batch_loss += loss_fn(pred_quantile, batch_y)
                    
                total_loss += batch_loss.item()
                
        return total_loss / len(dataloader)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            early_stopping_patience: int = 20) -> Dict[str, List[float]]:
        """
        Fit the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            early_stopping_patience: Early stopping patience
            
        Returns:
            Training history
        """
        # Create datasets
        train_dataset = ElectricityDemandDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = ElectricityDemandDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
        # Training loop
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate_epoch(val_loader)
                history['val_loss'].append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                    
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")
                    
        return history
        
    def predict(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features for prediction
            batch_size: Batch size for prediction
            
        Returns:
            Quantile predictions
        """
        self.model.eval()
        predictions = []
        
        dataset = ElectricityDemandDataset(X, np.zeros(len(X)))  # Dummy targets
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                batch_x = batch_x.to(self.device)
                pred = self.model(batch_x)
                predictions.append(pred.cpu().numpy())
                
        return np.vstack(predictions)