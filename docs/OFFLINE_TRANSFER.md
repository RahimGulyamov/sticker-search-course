# Ноутбук → YT → GPU-под без интернета

Использовать обновлённый `sticker_search_project.zip`: в нём есть новые скрипты
`laptop_to_yt.sh`, `pod_from_yt.sh`, модуль `offline_models` и `configs/offline.env`.
Команды рассчитаны на bash/zsh в macOS или Linux и Python 3.10–3.12.

Ноутбук должен иметь доступ к внешним источникам, Jupiter и рабочую YT-авторизацию.
Под должен иметь доступ к Jupiter. Веса моделей не зависят от ОС ноутбука;
бинарные Python-пакеты для пода зависят от его ОС, архитектуры и Python.

## 1. На ноутбуке: лёгкое окружение

Распаковать архив, открыть обычный локальный терминал ноутбука:

```bash
cd sticker_search
python3 -m venv .venv-transfer
source .venv-transfer/bin/activate
python -m pip install -r requirements/transfer.txt
source configs/pod.env
```

Здесь НЕ нужен `pip install -e .`: он установит ML-зависимости. Для скачивания
и переноса достаточно Pillow, huggingface-hub, pooch и ytsaurus-client.
Проверить YT до длинного скачивания:

```bash
python - <<'PY'
import os
import yt.wrapper as yt
c = yt.YtClient(proxy=os.environ['YT_PROXY'], token=os.environ.get('YT_TOKEN'))
print('YT доступен, корень проекта существует:', c.exists(os.environ['STICKER_YT_ROOT']))
PY
```

Результат `False` допустим: каталог ещё не создан. Ошибка сети/авторизации
означает, что надо настроить рабочее подключение и обычный YT-токен на ноутбуке.
Токен в архив и чат не передавать. Скрипты используют `YT_TOKEN` либо штатную
конфигурацию YT-клиента; uploader создаст каталог при наличии прав.

## 2. На ноутбуке: датасет и модели baseline

```bash
bash scripts/laptop_to_yt.sh dataset
bash scripts/laptop_to_yt.sh retrieval
```

Первая команда скачает исходный архив 6.6 GB и дополнительные картинки,
восстановит 30 914 изображений по сохранённым SHA-256, упакует и загрузит в YT.
Вторая скачает два энкодера поиска вместе с конфигами/токенизаторами и отправит
их отдельным набором. Ни модели, ни GPU на ноутбуке не запускаются.

| Содержимое | Путь на `YOUR_YT_PROXY` |
|---|---|
| Изображения и метаданные | `//home/USER/cv_project/datasets/v1` |
| CLIP и multilingual encoder | `//home/USER/cv_project/models/retrieval/v1` |
| Qwen для описаний/OCR и перевода запросов | `//home/USER/cv_project/models/annotation/v1` |
| FLUX, переводчик промптов и U2Net | `//home/USER/cv_project/models/generation/v1` |

При разрыве связи повторить ту же команду. Успешные скачанные файлы и уже
загруженные YT-файлы переиспользуются. Упаковка может выполниться повторно.
Отличающиеся существующие файлы YT не перезаписываются: нужна новая версия пути.
Готовность набора отмечает `bundle.json`, записываемый последним.

## 3. На поде: скачать из YT и включить офлайн-режим

Добавить обновлённые файлы проекта в рабочую папку пода через VS Code.
В терминале пода, из корня проекта, в уже установленном окружении:

```bash
bash scripts/pod_from_yt.sh dataset
bash scripts/pod_from_yt.sh retrieval
source configs/pod.env
source configs/offline.env
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
bash scripts/run_baseline.sh
```

Перед GPU-запуском нужна рабочая CUDA-сборка torch (`torch.cuda.is_available()`
должно быть `True`). Если сейчас стоит `+xpu`, выполнить восстановление окружения
из раздела 6. Датасет и веса можно переносить до исправления torch.

`configs/offline.env` выставляет пути кеша Hugging Face и U2Net, а также
`HF_HUB_OFFLINE=1` и `TRANSFORMERS_OFFLINE=1`. После открытия нового терминала
обе команды `source` повторить. Отсутствующий файл модели приведёт к ошибке
офлайн-кеша; внешнее скачивание не используется.

Результаты baseline: `runs/full_baseline_v1/dev/metrics.json` и `cases.jsonl`.

## 4. Передать остальные модели

Пока baseline запускается в поде, на ноутбуке можно передавать следующий набор:

```bash
bash scripts/laptop_to_yt.sh annotation
bash scripts/laptop_to_yt.sh generation
```

После завершения соответствующей загрузки, в поде:

```bash
bash scripts/pod_from_yt.sh annotation
bash scripts/pod_from_yt.sh generation
source configs/pod.env
source configs/offline.env
```

Далее выполнить пробную разметку 500 картинок по `POD_RUNBOOK.md`, затем полный
пайплайн. Веса для генерации включают U2Net ONNX: сам пакет rembg этих весов
не содержит. Для FLUX загружается только формат Diffusers, без второй копии
монолитного checkpoint. Дубли `.bin` не загружаются там, где есть safetensors.

## 5. Место на диске и проверка переноса

Для датасета и baseline ориентир — 25–30 GB свободного места. Для всех моделей,
исходных файлов и промежуточных tar одновременно понадобится значительно больше:
закладывать около 150–180 GB на устройстве, где всё хранится сразу. Это оценка
запаса, не измеренный размер окончательного набора. Downloader печатает точный
объём выбранных весов; упаковка добавляет примерно ещё одну копию файлов.

Модели упакованы потоково, без чтения многогигабайтного checkpoint целиком в RAM.
Каждый архив и восстановленный файл проверяются по SHA-256. HF-токены, кеш Xet,
служебные lock-файлы и настройки ноутбука в набор моделей не попадают.
Только закреплённые snapshot-файлы, конфиги и выбранные веса.

Локально проверены восстановление архивов, обнаружение порчи и защита от выхода
за каталог. Реальное подключение к вашему YT из этой сессии не выполнялось.
Qwen/FLUX необходимо проверить на целевых GPU после переноса.

## 6. Если CUDA-пакеты тоже требуется принести с ноутбука

Предыдущая онлайн-команда `repair_cuda.sh` в поде без внешнего доступа не
сработает. Сначала в поде получить параметры целевого Python и системы:

```bash
python - <<'PY'
import sys, platform, importlib.metadata as m
print('Python:', sys.version)
print('Platform:', platform.system(), platform.machine(), platform.libc_ver())
for name in ['torch', 'torchvision', 'torchaudio', 'triton', 'ytsaurus-client']:
    try: print(name, m.version(name))
    except m.PackageNotFoundError: print(name, 'not installed')
PY
nvidia-smi
```

Для офлайн-установки нужен wheelhouse под ЭТОТ Linux/Python, включая зависимости
CUDA-сборки. Обычный `pip download` на Mac выберет пакеты Mac; так делать нельзя.
Команда должна указывать целевые platform/python-version/implementation/abi
либо выполняться в соответствующем Linux-контейнере. Нельзя обещать готовый
wheelhouse без этих параметров. Пока можно выполнить весь перенос данных и весов.

Собранный wheelhouse переносится тем же механизмом:

```bash
# Ноутбук: после подготовки корректных wheels в data/wheelhouse
python -m sticker_search.file_bundle pack --input data/wheelhouse --output data/wheels_bundle
python -m sticker_search.yt_transfer upload --local data/wheels_bundle \
  --remote "$STICKER_YT_ROOT/environment/cu128_v1"

# Под:
python -m sticker_search.yt_transfer download --local data/wheels_bundle \
  --remote "$STICKER_YT_ROOT/environment/cu128_v1"
python -m sticker_search.file_bundle unpack --input data/wheels_bundle --output data/wheelhouse
```

Установка из wheelhouse использует `pip install --no-index --find-links=...`.
Удалять существующий torch до подготовки полного подходящего wheelhouse не нужно.

Документация:
- https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables
- https://pip.pypa.io/en/stable/cli/pip_download/
