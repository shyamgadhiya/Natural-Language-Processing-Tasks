import torch.nn as nn
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        x = self.embed(x) # (batch, seq_len, embed_dim)
        out, (h, c) = self.lstm(x) # out: (batch, seq_len, hidden_dim)
        return out