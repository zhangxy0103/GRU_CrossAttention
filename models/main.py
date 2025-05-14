from GRU_CrossAttention.models.cross_attention import CrossAttention
from GRU_CrossAttention.models.gur_cell import GRUCell
import torch.nn as nn
import torch

class CrossAttentionGRU(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim, v_dim, num_heads, hidden_size, output_dim=1):
        super(CrossAttentionGRU, self).__init__()
        self.cross_attention = CrossAttention(in_dim1, in_dim2, k_dim, v_dim, num_heads)
        self.gru_cell = GRUCell(in_dim1, hidden_size)
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x1, x2, hidden=None):
        if len(x1.size()) == 2:
            x1 = x1.unsqueeze(1)
        if len(x2.size()) == 2:
            x2 = x2.unsqueeze(1)

        batch_size, seq_len1, _ = x1.size()

        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x1.device)

        attn_output = self.cross_attention(x1, x2)

        outputs = []
        for t in range(seq_len1):
            hidden = self.gru_cell(attn_output[:, t, :], hidden)
            outputs.append(hidden.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)

        final_hidden = outputs[:, -1, :]
        logits = self.fc(final_hidden)

        return logits