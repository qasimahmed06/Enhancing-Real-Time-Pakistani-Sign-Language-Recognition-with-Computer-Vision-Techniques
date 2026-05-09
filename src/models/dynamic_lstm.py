import torch
import torch.nn as nn
from torchvision.models import Inception_V3_Weights, inception_v3


class Attention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        attention_weights = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_weights, dim=1)
        context = torch.sum(attention_weights * lstm_out, dim=1)
        return context


class DynamicLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        attention_dim: int = 256,
    ):
        super().__init__()
        
        self.cnn = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=False)
        
        for param in self.cnn.parameters():
            param.requires_grad = False
        
        cnn_out_dim = 2048
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        
        self.attention = Attention(hidden_dim, attention_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = frames.shape
        
        frames_flat = frames.view(batch_size * num_frames, channels, height, width)
        cnn_features = self.cnn(frames_flat)
        
        cnn_features = cnn_features.view(batch_size, num_frames, -1)
        
        lstm_out, _ = self.lstm(cnn_features)
        
        context = self.attention(lstm_out)
        
        logits = self.classifier(context)
        return logits


def build_dynamic_lstm(
    num_classes: int = 4,
    hidden_dim: int = 512,
    num_layers: int = 2,
    dropout: float = 0.3,
    attention_dim: int = 256,
) -> nn.Module:
    return DynamicLSTM(
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        attention_dim=attention_dim,
    )
