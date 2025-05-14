class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        # 定义层归一化
        self.layer_norm_x = nn.LayerNorm(input_size)
        self.layer_norm_h = nn.LayerNorm(hidden_size)

        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = self.layer_norm_x(x)
        hidden = self.layer_norm_h(hidden)

        # 保持 x 的维度
        batch_size = x.size(0)
        x = x.view(batch_size, -1)

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        # 切分门控输出
        i_r, i_i, i_n = gate_x.chunk(3, dim=1)
        h_r, h_i, h_n = gate_h.chunk(3, dim=1)

        # 使用 sigmoid 和 tanh 激活函数
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)

        # 计算新的隐藏状态
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)

        return hy