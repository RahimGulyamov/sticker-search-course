# Генерация и удаление фона: два независимых шага

В `app_siglip2.py` и `app.py` доступны две отдельные кнопки:

1. **Сгенерировать изображение**: Qwen переводит запрос, FLUX создаёт исходник.
   Фон пока остаётся; исходник можно посмотреть и скачать.
2. **Удалить фон**: U²-Net через rembg обрабатывает сохранённый исходник на CPU.
   Результат — отдельный PNG с прозрачностью. Повторное нажатие не запускает FLUX.

Исходник и результат сохраняются отдельно в `runs/generated/<id>/`.
Результаты остаются в интерфейсе при повторном выполнении виджетов текущей сессии.
При новой успешной генерации предыдущая маска убирается из интерфейса.
Файлы на диске при этом остаются. После перезагрузки страницы состояние сессии
может сброситься; исходники доступны в `runs/generated/`.

## Обновление на поде

Загрузи `generation_steps_v1.zip` в корень проекта и выполни:

```bash
cd /home/jovyan/sticker_search
python -m zipfile -e generation_steps_v1.zip .
source configs/pod.env
source configs/offline.env
export CUDA_VISIBLE_DEVICES=0
export STICKER_SIGLIP_RUN=/tmp/cv_project/siglip2_v1
export STICKER_ENABLE_GENERATION=1
python -m streamlit run app_siglip2.py \
  --server.address 127.0.0.1 --server.port 8501 --server.headless true \
  --browser.gatherUsageStats false
```

Перед запуском останови старый процесс Streamlit через Ctrl+C. Порт 8501
открывается через VS Code Ports. Обновление не требует переобучения,
новых весов или установки пакетов на поде.

## Python API

```python
from sticker_search.generate import generate_image
from sticker_search.remove_background import remove_background

# Новая папка на каждую генерацию: существующая не перезаписывается.
original = generate_image("кот с чашкой кофе", "runs/generated/my_cat")
# Можно посмотреть original.png и запустить второй шаг позже.
sticker = remove_background(original, "runs/generated/my_cat/sticker.png")
# Можно также обработать любое другое локальное изображение:
# sticker = remove_background("my_image.jpg", "my_image_sticker.png")
```

`generate_image` не загружает rembg/U²-Net. `remove_background` не использует
FLUX, Qwen, torch или GPU; ему нужны Pillow, rembg/ONNX Runtime и локальные
веса U²-Net (`U2NET_HOME`, уже заданный `configs/offline.env`).
Путь результата удаления фона должен отличаться от исходника и иметь `.png`.

## Терминал

```bash
# Только генерация. Директория результата должна быть новой.
python -m sticker_search.generate "кот с чашкой кофе" \
  --output runs/generated/my_cat --image-only

# Отдельное удаление фона; можно повторять без генерации.
python -m sticker_search.remove_background \
  runs/generated/my_cat/original.png \
  --output runs/generated/my_cat/sticker.png
```

Метаданные записываются отдельно: `image_generation.json` и
`sticker.background.json` (параметры, хеши, время каждого шага).
Постоянная alpha-маска считается ошибкой; исходник сохраняется для повторной
обработки. Непостоянная маска ещё не гарантирует хорошее отделение объекта.

Старый `generate(...)` и запуск `python -m sticker_search.generate` без
`--image-only` оставлены как совместимая последовательность двух функций.
Они дополнительно создают прежний `generation.json` для скрипта финальных проверок.
В интерфейсе используется только раздельный вариант.

## Проверки

```bash
python -m unittest discover -s tests -p 'test_generation_steps.py' -v
python -m unittest discover -s tests -p 'test_generation_ui.py' -v
```

Проверки используют подменённый инференс и не измеряют качество изображений.
Отдельно на поде нужно сгенерировать изображение, удалить фон и визуально
проверить обе картинки.
