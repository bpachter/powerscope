import torch, torch.nn as nn

class QuantileLSTM(nn.Module):
    def __init__(self, input_size, hidden, layers, dropout, quantiles=(0.1,0.5,0.9)):
        super().__init__()
        self.qs = sorted(list(quantiles))
        self.lstm = nn.LSTM(input_size, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Linear(hidden, len(self.qs))
    def forward(self, x):
        # x: [B, T, F]
        out,_ = self.lstm(x)
        last = out[:,-1,:]
        q = self.head(last)
        return q

def quantile_loss(pred, target, quantiles):
    # pred: [B, Q], target: [B]
    losses = []
    for i, q in enumerate(quantiles):
        e = target - pred[:, i]
        losses.append(torch.max(q*e, (q-1)*e).unsqueeze(1))
    return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
