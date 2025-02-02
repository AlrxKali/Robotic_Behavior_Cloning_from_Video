import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import DAMLConfig
from .model import DAMLModel

class DAMLTrainer:
    """Trainer class for DAML model"""
    def __init__(self, config: DAMLConfig):
        self.config = config
        self.model = DAMLModel(config).to(config.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> dict:
        """Train the model"""
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                batch = batch.to(self.config.device)
                loss = self._train_step(batch)
                train_loss += loss
                
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            
            if val_loader:
                val_loss = self._validate(val_loader)
                history['val_loss'].append(val_loss)
                print(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")
            else:
                print(f"Epoch {epoch+1}/{self.config.num_epochs} - Train Loss: {avg_train_loss:.4f}")
        
        return history
    
    def _train_step(self, batch: torch.Tensor) -> float:
        """Single training step"""
        self.optimizer.zero_grad()
        output = self.model(batch)
        loss = self.criterion(output[:, :-1], batch[:, 1:])
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.config.device)
                output = self.model(batch)
                loss = self.criterion(output[:, :-1], batch[:, 1:])
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def save_model(self, path: str):
        """Save model checkpoint"""
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'config': self.config}, path)
    
    def load_model(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
