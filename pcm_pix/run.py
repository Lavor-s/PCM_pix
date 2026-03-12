from __future__ import annotations

"""
run.py — управление "запусками" (runs) и артефактами.

Идея: каждый запуск ноутбука создаёт папку `outputs/<run_name>/` со структурой:
- logs/    : логи (и вывод в консоль)
- models/  : модели + скейлеры
- results/ : численные результаты/таблицы/solutions
- gds/     : экспорт GDS (когда подключим)

Так ноутбук остаётся тонким, а пути/логирование не размазаны по коду.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import logging


@dataclass(frozen=True)
class Run:
    root: Path
    logs: Path
    models: Path
    results: Path
    gds: Path
    logger: logging.Logger

def init_notebook_run(cfg, notebook_name: str, outputs_dir: str | Path = "outputs"):
    run = start_run(outputs_dir=outputs_dir, run_name=cfg["run_name"])
    logger = run.logger
    save_json(run.results / "config.json", cfg)
    logger.info("%s started", notebook_name)
    return run, logger

def _make_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger(f"pcm_run:{log_file.parent.parent.name}")
    logger.setLevel(logging.INFO)

    # важно: чтобы при повторном запуске ячейки не было дублей в логах
    logger.handlers.clear()

    # важно для Jupyter: иначе сообщения "всплывают" в root-logger и печатаются дважды
    logger.propagate = False

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
    models = root / "models"
    results = root / "results"
    gds = root / "gds"

    for p in (logs, models, results, gds):
        p.mkdir(parents=True, exist_ok=True)

    logger = _make_logger(logs / "run.log")
    logger.info("run_dir=%s", root.resolve())
    return Run(root=root, logs=logs, models=models, results=results, gds=gds, logger=logger)


def append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(text)


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")