import torch
import torch.nn as nn
import torch.nn.functional as F


class PSL_LSTM(nn.Module):
    """LSTM model with attention mechanism for sign language gesture recognition."""
    
    def __init__(self, input_size=512, hidden_size=256, num_layers=2, num_classes=4, dropout=0.5, bidirectional=True):
        super(PSL_LSTM, self).__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        self.attention = nn.Sequential(
            nn.Linear(lstm_output_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
    
    def forward(self, x, mask=None):
        lstm_out, _ = self.lstm(x)
        attention_weights = self.attention(lstm_out)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(attention_weights, dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        return self.classifier(attended)


if __name__ == "__main__":
    model = PSL_LSTM()
    x = torch.randn(8, 30, 512)
    mask = torch.ones(8, 30)
    output = model(x, mask)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")