class Config:
    def __init__(self):
        # data path
        self.data_paths = {
            'x1': 'path',
            'x2': 'path'
        }

        # 模型参数
        self.k_dim = 12
        self.v_dim = 12
        self.num_heads = 1
        self.hidden_size = 32

        # 训练参数
        self.batch_size = 32
        self.epochs = 1000
        self.lr = 0.0008
        self.weight_decay = 1e-4
        self.max_grad_norm = 0.5
        self.patience = 20

        # 其他
        self.seed = 42
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')