from __future__ import annotations
# pyright: reportMissingImports=false

"""
nn_surrogate.py — суррогатные нейросети (PyTorch) + сохранение/загрузка.

В проекте 2 суррогата:
- для аморфного состояния (N=0)
- для кристаллического состояния (N=1)

Модель предсказывает 4 значения: [Rcos, Rsin, Tcos, Tsin].

Поддерживаются 2 формата:
- "new" формат: state_dict + joblib-скейлеры (повторяемо/безопаснее)
- "legacy" формат: torch.save(model_object) и torch.save(scaler_object) из старого ноутбука
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

from pcm_pix.features import load_mesh_tables, make_nn_dataset
from pcm_pix.metrics import evaluate_surrogate


class Net(nn.Module):
    """Архитектура сети, надо бы как-нибудь выполнить гиперпараметрическую оптимизацию."""
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
    """
    Обёртка вокруг модели и скейлеров.

    predict() принимает A,D,B (в метрах, как в mesh таблицах)
    возвращает физические значения (после inverse_transform).

    """
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



def build_ANN(
    cfg: Dict[str, Any],
    run
):
    """
    Полный пайплайн:
    - выбрать device (cpu/cuda)
    - собрать датасет из mesh-таблиц
    - загрузить или обучить суррогаты (legacy/new)
    - посчитать QA-метрики качества

    Возвращает SimpleNamespace с полями:

    df      : полный DataFrame
    am_data : аморфный датасет для заданной длины волны
    cr_data : кристаллический датасет для заданной длины волны
    X_0     : признаки (a, d, b) аморфного датасета
    y_0     : цели (Rcos, Rsin, Tcos, Tsin) аморфного датасета
    X_1     : признаки (a, d, b) кристаллического датасета
    y_1     : цели (Rcos, Rsin, Tcos, Tsin) кристаллического датасет

    sur0    : суррогат аморфного датасета
    sur1    : суррогат кристаллического датасета
    qa_am   : QA-метрики аморфного датасета
    qa_cr   : QA-метрики кристаллического датасета

    """
    logger = getattr(run, "logger", None)

    mesh_tables = load_mesh_tables(cfg, base_dir=cfg["adb_data_dir"])
    df, data_0, data_1, X_0, y_0, X_1, y_1 = make_nn_dataset(mesh_tables, wl=cfg["wl"])
    sur0, sur1 = train_or_load_surrogates(X_0, y_0, X_1, y_1, run, cfg)

    logger.info("surrogates OK")
    logger.info("dataset: df=%s data_0=%s data_1=%s", len(df), len(data_0), len(data_1))
    logger.info("X_0=%s y_0=%s | X_1=%s y_1=%s", X_0.shape, y_0.shape, X_1.shape, y_1.shape)

    # QA-метрики
    qa_n = int(cfg.get("qa_n", 5000))
    qa_am = evaluate_surrogate(data_0, sur0, n=qa_n, label="am")
    qa_cr = evaluate_surrogate(data_1, sur1, n=qa_n, label="cr")

    if logger is not None:
        logger.info("QA am: %s", qa_am)
        logger.info("QA cr: %s", qa_cr)

    return [
        df,
        data_0,
        data_1,
        X_0,
        y_0,
        X_1,
        y_1,
        sur0,
        sur1,
        qa_am,
        qa_cr
    ]


def train_or_load_surrogates(X_0, y_0, X_1, y_1, run, cfg: Dict[str, Any]):
    """
    X_0     : признаки (a, d, b) аморфного датасета
    y_0     : цели (Rcos, Rsin, Tcos, Tsin) аморфного датасета
    X_1     : признаки (a, d, b) кристаллического датасета
    y_1     : цели (Rcos, Rsin, Tcos, Tsin) кристаллического датасет
    run: результат start_run(...)
    cfg: словарь с параметрами
    """
    device = cfg.get("device")
    epochs = int(cfg.get("epochs"))
    lr = float(cfg.get("lr"))
    ann_data_dir = Path(cfg.get("ann_data_dir"))
    train_load_mode = cfg.get("train_load_mode")

    tag0 = "sb2se3_am"
    tag1 = "sb2se3_cr"

    need_train = train_load_mode == "train"

    for tag in (tag0, tag1):
        if not (ann_data_dir / f"{tag}.pt").exists():
            need_train = True

    #load

    if not need_train:
        if hasattr(run, "logger"):
            run.logger.info("loading surrogates from %s", run.models)
        return _load(ann_data_dir, tag0, device), _load(ann_data_dir, tag1, device)

    #train

    if hasattr(run, "logger"):
        run.logger.info("training surrogates (device=%s, epochs=%s, lr=%s)", device, epochs, lr)

    sur0, m0 = _train_one(X_0, y_0, device=device, epochs=epochs, lr=lr, logger=getattr(run, "logger", None))
    _save(run.models, tag0, sur0.model, sur0.scaler_x, sur0.scaler_y)

    sur1, m1 = _train_one(X_1, y_1, device=device, epochs=epochs, lr=lr, logger=getattr(run, "logger", None))
    _save(run.models, tag1, sur1.model, sur1.scaler_x, sur1.scaler_y)

    if hasattr(run, "logger"):
        run.logger.info("saved models to %s", run.models)
        run.logger.info("am metrics=%s | cr metrics=%s", m0, m1)

    return sur0, sur1



def _load(run_models_dir: Path, tag: str, device: str) -> Surrogate:
    """Загрузка настроек"""
    model = Net()
    state = torch.load(run_models_dir / f"{tag}.pt", map_location="cpu")
    model.load_state_dict(state)

    scaler_x = joblib.load(run_models_dir / f"{tag}_scaler_x.pkl")
    scaler_y = joblib.load(run_models_dir / f"{tag}_scaler_y.pkl")
    return Surrogate(model=model, scaler_x=scaler_x, scaler_y=scaler_y, device=device)


def _save(run_models_dir: Path, tag: str, model: nn.Module, scaler_x: MinMaxScaler, scaler_y: MinMaxScaler) -> None:
    """Сохраняем модель в 'new' формате: state_dict + joblib scalers."""
    run_models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_models_dir / f"{tag}.pt")
    joblib.dump(scaler_x, run_models_dir / f"{tag}_scaler_x.pkl")
    joblib.dump(scaler_y, run_models_dir / f"{tag}_scaler_y.pkl")



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
    """Обучение одной сети + возврат базовых метрик train/test MSE."""
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
