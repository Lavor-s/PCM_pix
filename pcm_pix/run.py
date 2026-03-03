from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging


@dataclass(frozen=True)
class Run:
    root: Path
    logs: Path
    plots: Path
    models: Path
    results: Path
    gds: Path
    logger: logging.Logger


def _make_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"pcm_run:{log_file.parent.parent.name}")
    logger.setLevel(logging.INFO)

    # важно: чтобы при повторном запуске ячейки не было дублей в логах
    logger.handlers.clear()
    
    logger.propagate = False  # <-- ДОБАВЬ ЭТУ СТРОКУ

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def start_run(outputs_dir: str | Path = "outputs", run_name: str | None = None) -> Run:
    outputs_dir = Path(outputs_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = run_name or f"run_{ts}"
    root = outputs_dir / run_name

    logs = root / "logs"
    plots = root / "plots"
    models = root / "models"
    results = root / "results"
    gds = root / "gds"

    for p in (logs, plots, models, results, gds):
        p.mkdir(parents=True, exist_ok=True)

    logger = _make_logger(logs / "run.log")
    logger.info("run_dir=%s", root.resolve())
    return Run(root=root, logs=logs, plots=plots, models=models, results=results, gds=gds, logger=logger)


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")