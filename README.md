# PCM_pix

Проект `PCM_pix` — это **рефакторинг “большого” исследовательского ноутбука** в
воспроизводимую пайплайн-схему:
- загрузка материалов и табличных симуляций (mesh)
- обучение/загрузка суррогатных нейросетей (am/cr)
- оптимизация геометрии (PSO / DE / гибрид PSO→DE)
- сохранение всех артефактов в `outputs/<run_name>/`
- отдельный “графический” ноутбук для построения всех рисунков
- отдельный ноутбук для экспорта **GDS** (фабрикация)

## Быстрый старт

Минимальный сценарий (локально или на сервере):
- **`main_clean.ipynb`**: обучение/загрузка суррогатов + оптимизация + сохранение результатов
- **`plots.ipynb`**: построение графиков по сохранённым артефактам
- **`gds.ipynb`**: экспорт `.gds` и `.txt` по сохранённому `pos` (preset)

Если запускаешь на сервере без GUI, обычно удобно:

```bash
cd /path/to/PCM_pix
source /path/to/venv/bin/activate
jupyter nbconvert --execute main_clean.ipynb --to notebook --output main_clean.executed.ipynb
jupyter nbconvert --execute plots.ipynb --to notebook --output plots.executed.ipynb
```

## Структура данных (`data/`)

Входные файлы кладём в `data/`:
- **Материалы**:
  - `Sb2Se3_am.txt`, `Sb2Se3_cr.txt`
  - `Frantz-amorphous.csv`, `Frantz-crystal.csv`
- **Mesh-таблицы** (используются для датасета суррогата и “true vs pred” карт):
  - `Sb2Se3 - amorphous_mesh_sbse_2025.txt`
  - `Sb2Se3 - crystal_mesh12_sbse_2025.txt`
- **Legacy-модели (опционально)**:
  - `Sb2Se3_am_model_bagel_2025_updANN`, `Sb2Se3_am_scaler_X_bagel_2025_updANN`, `Sb2Se3_am_scaler_y_bagel_2025_updANN`
  - `Sb2Se3_cr_model_bagel_2025_updANN`, `Sb2Se3_cr_scaler_X_bagel_2025_updANN`, `Sb2Se3_cr_scaler_y_bagel_2025_updANN`
- **Доп. файлы для отдельных графиков**:
  - `EQ.txt`, `EQEHforV.txt` (multipole + field maps)
  - `Gauss_165_am.txt`, `Gauss_165_cr.txt` (derivative stack / neon overlay)

## Запуски и артефакты (`outputs/<run_name>/`)

Каждый запуск создаёт/использует папку `outputs/<run_name>/` со стандартной структурой:
- **`logs/`**: `run.log`
- **`models/`**: “new-format” суррогаты + скейлеры (если `surrogate_mode="new"`)
- **`results/`**:
  - `config.json` — сохранённый `CFG` (чтобы `plots.ipynb` мог работать без “состояния”)
  - `best_cost.txt`, `best_pos.txt` — результат PSO (если включено сохранение)
  - `best_de_cost.txt`, `best_de_pos.txt` — результат DE full
  - `de_progress.txt` — **прогресс DE** (пишется callback’ом; удобно при прерывании расчёта)
  - `solutions/` — каталог пресетов: `solutions/<preset_name>.json`
  - таблицы hyperopt (см. ниже)
- **`plots/`**: все PNG из `plots.ipynb`
- **`gds/`**: экспорт фабрикации (`.gds` + `.txt`) из `gds.ipynb`

## Основные ноутбуки

### `main_clean.ipynb` — расчётный ноутбук (главный)

Делает:
- загрузку данных из `data/`
- обучение/загрузку суррогатов
- оценку качества суррогатов
- оптимизацию выбранным режимом
- сохранение артефактов в `outputs/<run_name>/`

Главный вход — словарь **`CFG`** в начале ноутбука.

### `plots.ipynb` — графический ноутбук

Принципиально важно:
- **не обучает и не оптимизирует**
- читает **`CFG` из `outputs/<run_name>/results/config.json`** (если есть)
- строит графики, сохраняя их в `outputs/<run_name>/plots/`

В сложных графиках используются “слои” (layers), чтобы можно было включать/выключать наложения.

### `gds.ipynb` — экспорт GDS для фабрикации

Делает:
- загружает `pos` из `outputs/<run>/results/solutions/<preset>.json`
- формирует массивы `a/d/b`
- строит “решётку колец” и сохраняет:
  - `outputs/<run>/gds/<name>.gds`
  - `outputs/<run>/gds/<name>.txt` (таблица `a,d,b` + meta)

Экспорт реализован в модуле `pcm_pix/gds.py` и повторяет базовый сценарий из `3rd art`.

## Ключи конфигурации (`CFG`)

Ниже — **самые важные** ключи (полный список см. в `main_clean.ipynb`).

### Базовые
- **`run_name`**: имя папки в `outputs/`
- **`data_dir`**: где лежит папка `data/`
- **`wl`**: рабочая длина волны (метры), напр. `1.55e-6`
- **`mesh_names`**: список mesh-таблиц (am/cr)

### Суррогаты
- **`surrogate_mode`**: `"legacy" | "new"`
- **`device`**: `"cpu" | "cuda" | "auto"`
- **`epochs`**, **`lr`**: параметры обучения (если `new`)

### Геометрия и ограничения
- **`Nn`**: число “уровней/полос” (обычно 11)
- **`b_min_m`**: если `b < b_min_m` → считаем `b = 0` (как в исходном ноутбуке)

### Режим оптимизации
- **`opt_mode`**: один из
  - `"preset"`: просто загрузить пресет `pos` (без оптимизации)
  - `"pso"`: один прогон PSO
  - `"pso_until"`: повторять PSO с рестартами до достижения порога
  - `"de_full"`: полный DE (с init_ar/callback/constraints как в `to_server_arch`)
  - `"hybrid_pso_de"`: PSO → затем DE full (доработка)
- **`preset_name`**: имя пресета в `results/solutions/<preset_name>.json`

### PSO параметры (если `opt_mode` включает PSO)
- **`pso_n_particles`**, **`pso_iters`**
- **`pso_c1`**, **`pso_c2`**, **`pso_w`**
- **`pso_threshold`**, **`pso_max_restarts`** (для режима `pso_until`)

### DE параметры (для `de_full` / `hybrid_pso_de`)
- **`de_init_mode`**: `"init_ar" | "x0"`
  - `"init_ar"`: строится “init-array” популяции вокруг `pos` (см. `make_init_ar_from_pos`)
  - `"x0"`: `pos` передаётся как `x0`, а `init` остаётся стандартным (LHS)
- **`de_init_N`**: размер init-array (для `init_ar`)
- **`de_mutation`**, **`de_recombination`**
- **`de_maxiter`**, **`de_popsize`**
- **`de_tol`**, **`de_atol`**, **`de_polish`**
- **`de_callback_every`**: как часто писать прогресс в `results/de_progress.txt`
- **`de_use_linear_constraint`**: включить линейные ограничения (аналог закомментированного блока в 3rd art)

### Hyperopt (подбор гиперпараметров оптимизаторов)

Результаты hyperopt кешируются в `outputs/<run_name>/results/`.

Ключи:
- **`pso_hyperopt_mode`**: `"auto" | "use_saved" | "run"`
- **`de_hyperopt_mode`**: `"auto" | "use_saved" | "run"`

Файлы:
- **PSO**: `pso_random_search.csv`, `pso_refine.csv`
- **DE**: `de_random_search.csv`, `de_refine.csv`

## Как сохранить/использовать “пресеты” (`pos`)

Решения хранятся как JSON в:
- `outputs/<run_name>/results/solutions/<name>.json`

Формат:
- `pos`: список float
- `cost`: float (может быть `null`)
- `meta`: словарь (что угодно: дата, параметры, комментарии)

Полезные функции:
- `pcm_pix.optimize.save_solution(run, name, pos, cost, meta)`
- `pcm_pix.optimize.load_solution(path) -> (pos, cost, meta)`

## Экспорт GDS

Модуль: `pcm_pix/gds.py`  
Ноутбук: `gds.ipynb`

Ключи:
- **`gds_name`**: имя файлов
- **`gds_l_m`**: ширина полосы `l` (по умолчанию `20e-6`)
- **`gds_L_m`**: высота `L` (по умолчанию `11*l`)
- **`gds_layer_mode`**: `"fab"` (слой 0) или `"lum"` (слой зависит от `num_lay`)

Примечание: нужен пакет **`gdspy`** (установлен не всегда). Если его нет — `pcm_pix.gds`
выдаст понятную ошибку, что нужно поставить зависимость.

## Архив/наследие

Файлы:
- `3rd art_PCM_bagel_2025.ipynb`
- `to_server_arch-Copy1.ipynb`

Они оставлены как **источник истины** по старой логике и как “архив”, но актуальная работа
ведётся через `main_clean.ipynb` + `plots.ipynb` + `gds.ipynb`.

