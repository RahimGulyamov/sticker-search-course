# SigLIP 2: полноценное дообучение энкодеров

Новый эксперимент после CLIP + малого адаптера. GPU-эксперимент выполнен: выбрана эпоха 2 по dev.
Результаты и ограничения опубликованы в [README](../README.md) и
[evidence/siglip2](../evidence/siglip2/). Ниже сохранены команды и план запуска.
Предыдущие результаты, test и PDF остаются снимком эксперимента full_v2.

## Что обучается

- Модель: `google/siglip2-so400m-patch14-384`, revision
  `e8e487298228002f3d8a82e0cd5c8ea9c567f57f`, Apache-2.0.
- Две Transformer-башни: по 27 слоёв, hidden size 1152, изображения 384×384,
  patch 14. Около 1,136 млрд параметров суммарно. HF `model_type=siglip`
  здесь корректен: это fixed-resolution SigLIP 2, а не вариант NaFlex.
- Обновляются **все параметры обеих башен**, включая эмбеддинги,
  проекции, logit scale и bias. Нет residual MLP-адаптера, нет замороженных
  предварительно вычисленных признаков в обучении.
- FP32 параметры/AdamW, BF16 autocast, SDPA, gradient checkpointing.
- Стартовые параметры: 4 GPU × 32 пары = 128 пар на шаг, 3 эпохи,
  learning rate 5e-6, warmup 5%, cosine decay, grad clip 1.0.
  Это стартовая конфигурация, не результат подбора гиперпараметров.
- Исходная модель оценивается как эпоха 0. Выбор: среднее EN/RU **dev NDCG@5**
  по image-only поиску. Если обучение ухудшит результат, selected останется
  исходной моделью; это явно отражается в `complete.json`.

## Данные и loss

Каталог: 30 914 картинок. Разбиение по группам: train 21 805, dev 4 469,
test 4 640. В поиске всегда участвует весь каталог. Для градиентов используются
только train-пары и train-картинки; dev/test не входят в отрицательные примеры.

В train входят:

1. Каждая доступная допустимая EN/RU **визуальная подпись** к train-картинке,
   вес строки 0,5. Два языка дают две пары для одного изображения.
2. Исходные пользовательские EN-запросы, вес 1,0.
3. Их RU-переводы, вес 0,7: они не являются независимой человеческой разметкой.

`usage_queries_ru` не используются. Caption/OCR не генерируются повторно.
OCR используется в отдельном варианте поиска, см. ниже.
Текущие 39 пропущенных аннотаций и отфильтрованные подписи не заполняются
вымышленными таргетами. Изображение без валидного текста и без source query
остаётся в поисковом каталоге, но не получает обучающей пары. Доля покрытия
должна быть не ниже 98%; точные числа и ID сохраняются в `prepared/coverage.json`.
В выполненном запуске покрытие составило 21 776 из 21 805 train-картинок
(99,87%), подготовлено 49 318 пар. Автоматический фильтр выявляет
пустые/повторяющиеся ответы, но не гарантирует семантическую точность.

Loss — взвешенный multi-positive sigmoid. Для запроса i и изображения j:

`logit_ij = exp(logit_scale) * cosine(text_i, image_j) + logit_bias`.

Суммируем `softplus(-logit)` по известным положительным и `softplus(logit)` по
допустимым отрицательным парам; нормируем по сумме весов текстовых строк.
Изображения собираются со всех GPU через **differentiable all_gather**.
Множитель `world_size / sum(global_weights)` компенсирует усреднение DDP.
Повторяющиеся image-столбцы имеют суммарный вес 1. Известные positives
объединяются для одинакового токенизированного текста, включая коллизии после
обрезания. Неизвестные совпадения из той же image-group маскируются.
Это не решает проблему всех семантически похожих, но неразмеченных positives.

Негативы — **текущий общий батч**, а не все 30 тыс. одновременно. За эпоху
проходятся все подготовленные пары; это устраняет прежнее ограничение 405
supervised-картинками. DistributedSampler может дополнить конец эпохи максимум
на `world_size−1` строки; число записывается в `training_data.json`.

Текст приводится к нижнему регистру и дополняется до 64 токенов по рецепту
checkpoint. Число обрезанных текстов записывается. Alpha композируется на белый
фон. Случайный crop/flip не используется, чтобы не уничтожать надписи и жесты.

## 1. Обновить файлы на ноутбуке и поде

Скопировать `siglip2_training_v1.zip` в корень проекта на **обоих** устройствах:

```bash
python -m zipfile -e siglip2_training_v1.zip .
```

Архив содержит код и конфигурацию, без весов и без изменений `runs/full_v2`.
Существующую CUDA-сборку torch на поде **не переустанавливать**.
Новые пакеты для этого эксперимента в уже настроенном поде не нужны:
используются torch 2.9.1+cu128, transformers 4.57.1, sentencepiece 0.2.1,
safetensors 0.6.2 и остальные текущие зависимости проекта.

## 2. Ноутбук с интернетом: скачать и загрузить в YT

В корне проекта, в прежнем `.venv-transfer`:

```bash
source .venv-transfer/bin/activate
caffeinate -i bash scripts/siglip2_assets.sh laptop
```

`caffeinate` — команда macOS; она не даёт ноутбуку уснуть во время работы.
Скачивается **только SigLIP 2**, около 4,6 GB; FLUX/Qwen/датасет не скачиваются
повторно. Для весов и архива вместе нужно примерно 10 GB, лучше 12 GiB свободно.
Доступ к gated-репозиторию и HF login для этой модели не требуются.
Скрипт снимает offline-переменные только внутри своего процесса.
Используется прежний YT_TOKEN и адрес:

`//home/USER/cv_project/models/siglip2/v1`

После обрыва повторить ту же команду: HF использует локальные файлы, YT проверяет
checksums уже загруженных файлов. Отдельные `.part` файлы не удалять вручную.

## 3. Под без интернета: восстановить веса и запустить

В терминале VS Code на поде:

```bash
cd /home/jovyan/sticker_search
bash scripts/siglip2_assets.sh pod
mkdir -p runs/logs
set -o pipefail
bash scripts/run_siglip2.sh 2>&1 | tee runs/logs/siglip2_v1.log
```

Веса и большие checkpoints по умолчанию размещаются **в `/tmp/cv_project`**, а
не на заполненном `/home/jovyan`. Перед обучением проверяется наличие 60 GiB
свободного места для состояния AdamW, промежуточной атомарной записи `last.pt`
и выбранных весов. Это резерв, а не измеренный размер обучения. `/tmp` может
исчезнуть при удалении/пересоздании пода — сохранить результат в YT до этого.

Последовательность одного запуска:

1. CUDA/disk preflight и чтение всех изображений, проверка split и покрытия.
2. Два обучающих шага на четырёх GPU в отдельном `smoke/`.
3. Проверка ненулевых градиентов и изменения весов **обеих** башен; измерение VRAM.
4. Оценка исходного SigLIP 2 на EN/RU dev, весь каталог 30 914.
5. Три эпохи полноценного обучения; оценка на dev после каждой эпохи.
6. Сохранение лучшей модели, индекса и небольшого архива результатов.

Baseline-индексация может занимать заметное время; первые training loss появятся
после её окончания. Точное время и допустимый batch зависят от пода, заранее
не измерены. Test автоматически не запускается.

## Продолжение после остановки и OOM

Повторить `bash scripts/run_siglip2.sh` с теми же параметрами. Состояние
сохраняется каждые 500 шагов и после каждой эпохи. Восстанавливаются модель,
AdamW, scheduler, позиция в эпохе, RNG, история и лучший dev checkpoint.
Несохранённые шаги будут повторены. Уже завершённый запуск не обучается заново.
При resume должны совпадать данные, код обучения, конфигурация и число GPU.

Если **двухшаговый smoke** завершился CUDA OOM, выбрать новый run и уменьшить
batch; не менять torch, не удалять прежний run:

```bash
python - <<'PY'
import json
from pathlib import Path
p=Path('configs/siglip2_full.json')
c=json.loads(p.read_text())
c['batch_size_per_gpu']=16
Path('configs/siglip2_bs16.json').write_text(json.dumps(c,indent=2)+'\n')
PY
export STICKER_SIGLIP_CONFIG=configs/siglip2_bs16.json
export STICKER_SIGLIP_RUN=/tmp/cv_project/siglip2_bs16
bash scripts/run_siglip2.sh 2>&1 | tee runs/logs/siglip2_bs16.log
```

Global batch станет 64, то есть изменится набор отрицательных пар. Обычное
накопление градиентов не увеличивает число контрастивных негативов, поэтому
оно не используется как скрытая замена большого батча.

Переопределения: `STICKER_SIGLIP_ASSETS`, `STICKER_SIGLIP_MODEL`,
`STICKER_SIGLIP_RUN`, `STICKER_SIGLIP_CONFIG`, `SIGLIP_GPU_WORKERS`.
На другом числе GPU запускать отдельный experiment; полный корпус сохраняется.

## Что прислать для анализа

После завершения появится **`runs/siglip2_v1_results.zip`**: конфигурация,
покрытие, история, EN/RU метрики каждой эпохи, top-10 по запросам и HTML
сравнения до/после. Веса в этот небольшой архив не входят. В HTML явно указано,
что отобраны крайние изменения метрики; это не независимая оценка человеком.

Основные файлы внутри run:

- `prepared/coverage.json` — уникальные train-картинки и пропуски;
- `training_data.json` — число обучаемых параметров, пары, global batch;
- `smoke/complete.json` — градиенты обеих башен и peak VRAM каждой GPU;
- `history.jsonl` — исходная модель и все эпохи, отдельные EN/RU показатели;
- `selection.json` — выбранные веса/индекс, эпоха и dev-метрика;
- `last.pt` — полное состояние для продолжения (содержит доверенный Python state);
- `complete.json` — завершение и факт улучшения относительно исходной модели.

## Поиск, OCR и демо

```bash
source configs/pod.env
source configs/offline.env
python -m sticker_search.siglip.search --query 'весёлый кот с чашкой кофе'
streamlit run app_siglip2.py --server.address 0.0.0.0 --server.port 8501
```

Открыть порт 8501 через VS Code Ports. В демо доступны исходная и выбранная
модели. Поиск — нормированные векторы 1152, точное скалярное произведение по
каталогу. При 30 тыс. кандидатов ANN не нужен для этой реализации.
Опция надписей добавляет отдельную OCR TF-IDF ветвь через RRF (k=60).
Она не использовалась для выбора энкодера и должна оцениваться как отдельная
абляция на dev:

```bash
python -m sticker_search.siglip.evaluate --run /tmp/cv_project/siglip2_v1 \
  --which selected --split dev --method image_ocr_rrf
```

Для уже подготовленной генерации перед Streamlit задать
`export STICKER_ENABLE_GENERATION=1`. Генерация FLUX и удаление фона U2Net/rembg
разделены на две кнопки: «Сгенерировать изображение» и «Удалить фон».
Исходник можно посмотреть и скачать до удаления фона. Повторное удаление
не запускает FLUX. Отдельные Python API и CLI описаны в
[GENERATION_STEPS.md](GENERATION_STEPS.md).
Запускать поиск/генерацию после обучения, чтобы не конкурировать
за GPU. Сходство не является калиброванной вероятностью: автоматический порог
«подходящих стикеров нет» здесь не заявляется; генерация вызывается кнопкой.

## Финальная оценка и сохранение

После выбора параметров только по dev:

```bash
python -m sticker_search.siglip.evaluate --run /tmp/cv_project/siglip2_v1 \
  --which baseline --split test --allow-test
python -m sticker_search.siglip.evaluate --run /tmp/cv_project/siglip2_v1 \
  --which selected --split test --allow-test
python scripts/collect_siglip2_results.py --run /tmp/cv_project/siglip2_v1
```

Старый test уже просматривался в предыдущем эксперименте: новая оценка на нём
не является полностью новым слепым holdout. Не подбирать по нему параметры;
для сильного вывода нужна дополнительная независимая разметка русских запросов.
Метрики known-positive недооценивают неразмеченные подходящие альтернативы.
Test не заменяет разбор визуальных примеров и открыток.

Перед удалением пода можно сохранить весь run (включая AdamW) в отдельную YT-версию:

```bash
source configs/pod.env
python -m sticker_search.file_bundle pack --input /tmp/cv_project/siglip2_v1 \
  --output /tmp/cv_project/siglip2_v1_backup
python -m sticker_search.yt_transfer upload --local /tmp/cv_project/siglip2_v1_backup \
  --remote "$STICKER_YT_ROOT/experiments/siglip2/v1"
```

Это дополнительная большая копия, ей нужно свободное место. После изменения run
использовать новый versioned YT-каталог. Предобученные веса отдельно сохранены
в `models/siglip2/v1`. Метаданные содержат абсолютные пути: для прямого resume
восстанавливать те же пути `/tmp/cv_project/...` и расположение проекта.

## Проверки кода и источники

CPU integration использует **маленький случайный SiglipModel**, без скачивания
больших весов: подготовка, обновление обеих башен, оценка, сохранение и точное
продолжение после 3-го шага. Это проверка кода, не качества проекта.
Численно проверены loss, duplicate masking и масштабирование градиентов при
разбиении общего батча. В среде подготовки запрещены Gloo network sockets;
настоящий multi-process/NCCL и 4×A100 проверяются smoke-запуском на поде.

```bash
python -m pytest -q tests/test_siglip2.py
# Дополнительная проверка реального DDP в среде с доступными Gloo-сокетами:
STICKER_TEST_DDP=1 python -m pytest -q tests/test_siglip2.py
PYTHONPATH=. python tests/siglip2_integration.py /tmp/siglip2-code-test
```

- [Model card](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [SigLIP 2 paper](https://arxiv.org/abs/2502.14786)
- [Sigmoid loss paper](https://arxiv.org/abs/2303.15343)
- [Transformers 4.57.1 implementation](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/siglip/modeling_siglip.py)
- Курс: лекция 6, CLIP/SigLIP/SigLIP 2; лекция 5, metric learning/негативы.

Генеративные и self-distillation auxiliary losses первоначального pretraining
SigLIP 2 не воспроизводятся: дообучается готовый checkpoint с retrieval loss.
