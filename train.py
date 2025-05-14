import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
from models.main import CrossAttentionGRU
from utils.early_stopping import EarlyStopping
from utils.metrics import calculate_metrics
from data.load_data import load_and_preprocess_data
from config import Config


def train_and_validate():
    cfg = Config()

    x1, x2, y = load_and_preprocess_data(cfg.data_paths)

    # 10折交叉验证
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=cfg.seed)
    all_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x1, y)):
        print(f"\n=== Fold {fold + 1}/10 ===")

        # 数据分割
        train_set = Subset(list(zip(x1, x2, y)), train_idx)
        val_set = Subset(list(zip(x1, x2, y)), val_idx)

        # 初始化模型
        model = CrossAttentionGRU(
            in_dim1=x1.shape[1],
            in_dim2=x2.shape[1],
            k_dim=cfg.k_dim,
            v_dim=cfg.v_dim,
            num_heads=cfg.num_heads,
            hidden_size=cfg.hidden_size
        ).to(cfg.device)

        early_stopping = EarlyStopping(patience=cfg.patience, verbose=True)

        for epoch in range(cfg.epochs):
            train_metrics = train_epoch(model, train_set, cfg)
            val_metrics = validate_epoch(model, val_set, cfg)

            early_stopping(val_metrics['loss'], model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

        # 保存最佳模型
        torch.save(model.state_dict(), f"models/fold_{fold}_best.pt")
        all_metrics.append(val_metrics)

    print("\n=== Cross-validation Results ===")
    for metric in all_metrics[0].keys():
        avg = np.mean([m[metric] for m in all_metrics])
        print(f"Average {metric}: {avg:.4f}")


def train_epoch(model, dataset, cfg):
    model.train()
    total_loss = 0
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)

    for x1, x2, y in loader:
        x1, x2, y = x1.to(cfg.device), x2.to(cfg.device), y.to(cfg.device)

        # 训练步骤
        optimizer.zero_grad()
        outputs = model(x1, x2)
        loss = criterion(outputs, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return {'loss': total_loss / len(loader)}


def validate_epoch(model, dataset, cfg):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=cfg.batch_size)
        for x1, x2, y in loader:
            x1, x2, y = x1.to(cfg.device), x2.to(cfg.device), y.to(cfg.device)
            outputs = model(x1, x2)

            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return calculate_metrics(np.array(all_labels), np.array(all_preds))


if __name__ == "__main__":
    train_and_validate()