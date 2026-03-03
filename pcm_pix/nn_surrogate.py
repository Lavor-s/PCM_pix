from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib


def get_device() -> str:
    if torch.cuda.is_available():
        try:
            x = torch.tensor([1.0], device="cuda")
            y = x * 2
            _ = y.item()
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc2_5 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 4)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc2_5(x))
        x = self.fc3(x)
        return x


@dataclass
class Surrogate:
    model: nn.Module
    scaler_x: MinMaxScaler
    scaler_y: MinMaxScaler
    device: str = "cpu"

    def predict(self, A, D, B) -> np.ndarray:
        X = np.column_stack((A, D, B))
        Xs = self.scaler_x.transform(X)

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():
            y_scaled = self.model(
                torch.tensor(Xs, dtype=torch.float32, device=self.device)
            ).cpu().numpy()

        return self.scaler_y.inverse_transform(y_scaled)


def _save(run_models_dir: Path, tag: str, model: nn.Module, scaler_x: MinMaxScaler, scaler_y: MinMaxScaler) -> None:
    run_models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_models_dir / f"{tag}.pt")
    joblib.dump(scaler_x, run_models_dir / f"{tag}_scaler_x.pkl")
    joblib.dump(scaler_y, run_models_dir / f"{tag}_scaler_y.pkl")


def _load(run_models_dir: Path, tag: str, device: str) -> Surrogate:
    model = Net()
    state = torch.load(run_models_dir / f"{tag}.pt", map_location="cpu")
    model.load_state_dict(state)

    scaler_x = joblib.load(run_models_dir / f"{tag}_scaler_x.pkl")
    scaler_y = joblib.load(run_models_dir / f"{tag}_scaler_y.pkl")
    return Surrogate(model=model, scaler_x=scaler_x, scaler_y=scaler_y, device=device)


def _train_one(
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    epochs: int,
    lr: float,
    feature_range: Tuple[float, float] = (-1, 1),
    test_size: float = 0.2,
    random_state: int = 42,
    log_every: int = 200,
    logger=None,
) -> Tuple[Surrogate, Dict[str, float]]:
    scaler_x = MinMaxScaler(feature_range=feature_range)
    scaler_y = MinMaxScaler(feature_range=feature_range)

    Xs = scaler_x.fit_transform(X)
    ys = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=test_size, random_state=random_state
    )

    model = Net().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch % log_every == 0) or (epoch == epochs - 1):
            model.eval()
            with torch.no_grad():
                test_loss = criterion(model(X_test_t), y_test_t).item()
            if logger is not None:
                logger.info("epoch=%s train_loss=%.6f test_loss=%.6f", epoch, loss.item(), test_loss)

    sur = Surrogate(model=model, scaler_x=scaler_x, scaler_y=scaler_y, device=device)
    metrics = {"train_loss": float(loss.item()), "test_loss": float(test_loss)}
    return sur, metrics


def train_or_load_surrogates(ds, run, cfg: Dict[str, Any], force_train: bool = False):
    """
    ds: результат make_nn_dataset(...)
    run: результат start_run(...)
    cfg: словарь с параметрами (epochs, lr, device)
    """
    device = cfg.get("device", "cpu")
    epochs = int(cfg.get("epochs", 2000))
    lr = float(cfg.get("lr", 1e-3))

    tag0 = "sb2se3_am"
    tag1 = "sb2se3_cr"

    need_train = force_train
    for tag in (tag0, tag1):
        if not (run.models / f"{tag}.pt").exists():
            need_train = True

    if not need_train:
        if hasattr(run, "logger"):
            run.logger.info("loading surrogates from %s", run.models)
        return _load(run.models, tag0, device), _load(run.models, tag1, device)

    if hasattr(run, "logger"):
        run.logger.info("training surrogates (device=%s, epochs=%s, lr=%s)", device, epochs, lr)

    sur0, m0 = _train_one(ds.X_0, ds.y_0, device=device, epochs=epochs, lr=lr, logger=getattr(run, "logger", None))
    _save(run.models, tag0, sur0.model, sur0.scaler_x, sur0.scaler_y)

    sur1, m1 = _train_one(ds.X_1, ds.y_1, device=device, epochs=epochs, lr=lr, logger=getattr(run, "logger", None))
    _save(run.models, tag1, sur1.model, sur1.scaler_x, sur1.scaler_y)

    if hasattr(run, "logger"):
        run.logger.info("saved models to %s", run.models)
        run.logger.info("am metrics=%s | cr metrics=%s", m0, m1)

    return sur0, sur1


def _torch_load_compat(path: Path, map_location: str = "cpu"):
    # torch 2.x иногда имеет weights_only, иногда нет/ведёт себя по-разному
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_legacy_surrogates(cfg: Dict[str, Any], device: str = "cpu") -> tuple[Surrogate, Surrogate]:
    """
    Грузит старые артефакты, сохранённые через torch.save(model, ...) и torch.save(scaler, ...).
    Ожидает, что файлы лежат в cfg["legacy_dir"] (по умолчанию 'data').
    """
    legacy_dir = Path(cfg.get("legacy_dir", "data"))

    am_model_path = legacy_dir / cfg.get("am_model_file", "Sb2Se3_am_model_bagel_2025_updANN")
    am_sx_path = legacy_dir / cfg.get("am_scaler_x_file", "Sb2Se3_am_scaler_X_bagel_2025_updANN")
    am_sy_path = legacy_dir / cfg.get("am_scaler_y_file", "Sb2Se3_am_scaler_y_bagel_2025_updANN")

    cr_model_path = legacy_dir / cfg.get("cr_model_file", "Sb2Se3_cr_model_bagel_2025_updANN")
    cr_sx_path = legacy_dir / cfg.get("cr_scaler_x_file", "Sb2Se3_cr_scaler_X_bagel_2025_updANN")
    cr_sy_path = legacy_dir / cfg.get("cr_scaler_y_file", "Sb2Se3_cr_scaler_y_bagel_2025_updANN")

    model_0 = _torch_load_compat(am_model_path, map_location="cpu")
    scaler_X_0 = _torch_load_compat(am_sx_path, map_location="cpu")
    scaler_y_0 = _torch_load_compat(am_sy_path, map_location="cpu")

    model_1 = _torch_load_compat(cr_model_path, map_location="cpu")
    scaler_X_1 = _torch_load_compat(cr_sx_path, map_location="cpu")
    scaler_y_1 = _torch_load_compat(cr_sy_path, map_location="cpu")

    # Важно: модель может быть сохранена уже с архитектурой внутри, так что Net() не нужен
    sur0 = Surrogate(model=model_0, scaler_x=scaler_X_0, scaler_y=scaler_y_0, device=device)
    sur1 = Surrogate(model=model_1, scaler_x=scaler_X_1, scaler_y=scaler_y_1, device=device)
    return sur0, sur1