"""Build the stage-1 report exclusively from saved, measured experiment outputs."""
from pathlib import Path
import json
from collections import Counter
from html import escape

from PIL import Image as PILImage,ImageOps,ImageDraw
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle,Image

ROOT=Path(__file__).resolve().parents[1]
stats=json.loads((ROOT/'evidence/catalog/summary.json').read_text())
source_stats=json.loads((ROOT/'evidence/full_catalog/audit/summary.json').read_text())
metrics=json.loads((ROOT/'runs/cpu_evaluation/metrics.json').read_text())
config=json.loads((ROOT/'runs/cpu_adapter/config.json').read_text())
training=json.loads((ROOT/'runs/cpu_adapter/summary.json').read_text())
items=[json.loads(x) for x in (ROOT/'data/catalog/manifest.jsonl').read_text().splitlines()]
font_dir=Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DejaVu',str(font_dir/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVu-Bold',str(font_dir/'DejaVuSans-Bold.ttf')))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu',boldItalic='DejaVu-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyRU',fontName='DejaVu',fontSize=10.1,leading=15,spaceAfter=9,textColor=colors.HexColor('#26334b')))
styles.add(ParagraphStyle(name='TitleRU',fontName='DejaVu-Bold',fontSize=25,leading=30,spaceAfter=18,textColor=colors.HexColor('#17223c')))
styles.add(ParagraphStyle(name='HeadRU',fontName='DejaVu-Bold',fontSize=17,leading=22,spaceAfter=14,textColor=colors.HexColor('#17223c')))
styles.add(ParagraphStyle(name='SubRU',fontName='DejaVu-Bold',fontSize=11.5,leading=16,spaceBefore=8,spaceAfter=7,textColor=colors.HexColor('#374bcc')))
styles.add(ParagraphStyle(name='SmallRU',fontName='DejaVu',fontSize=8.5,leading=12,spaceAfter=7,textColor=colors.HexColor('#526078')))
styles.add(ParagraphStyle(name='CellRU',fontName='DejaVu',fontSize=9,leading=12,spaceAfter=0,textColor=colors.HexColor('#26334b')))
story=[]


def para(text,style='BodyRU'):
    story.append(Paragraph(text,styles[style]))


def head(n,title):
    if story:story.append(PageBreak())
    para(f'ЭТАП 1 / {n:02d}','SmallRU');para(title,'HeadRU')


def table(headers,rows,widths):
    data=[[Paragraph('<b>'+escape(str(x))+'</b>',styles['CellRU']) for x in headers]]
    data += [[Paragraph(escape(str(x)),styles['CellRU']) for x in row] for row in rows]
    t=Table(data,colWidths=[w*cm for w in widths],repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7ebff')),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7fb')]),('LEFTPADDING',(0,0),(-1,-1),8),
        ('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    story.append(t)


def figure(path,width_cm):
    with PILImage.open(path) as im: w,h=im.size
    story.append(Image(str(path),width=width_cm*cm,height=width_cm*cm*h/w,hAlign='LEFT'))
    story.append(Spacer(1,8))


head(1,'Постановка задачи и требования')
para('Интеллектуальный подбор<br/>стикеров и открыток','TitleRU')
para('Промежуточный отчёт по проекту курса компьютерного зрения. Обновление: 27 сентября 2026 года.','SmallRU')
para('Система получает свободный русскоязычный запрос и возвращает изображения, подходящие для отправки в заданной ситуации. Например, «поддержать друга перед экзаменом» задаёт намерение и тон; «кот с кофе» - визуальный объект; «с днём рождения без надписи» - повод и явное ограничение.')
table(['Требование','Критерий проверки'],[
 ('Поиск стикеров и открыток','Top-5; фильтр типа изображения; несколько допустимых ответов'),
 ('Учёт визуального смысла и надписи','Сравнение image-only, OCR и гибрида; разбор ошибок текста'),
 ('Русские запросы','Мультиязычный текстовый энкодер; отдельная RU-оценка на этапе 2'),
 ('Воспроизводимость','Фиксированные ревизии, seed=42, manifest и SHA-256'),
 ('Скорость','Проектная цель p95 полного поиска <1 с; измерить на целевой машине'),
 ('Генерация при неудовлетворительном поиске','Готовая генеративная модель, автоматическое удаление фона и RGBA PNG на этапе 2'),
],[5,12])
para(f'<b>Фактически выполнено:</b> собран каталог из {stats["images"]:,} изображений, проведены проверка файлов, дедупликация, предварительный групповой split и EDA. На отдельном пилоте из 472 изображений рассчитаны нейросетевые признаки, проведены 5 эпох обучения и оценка. Полная GPU-разметка и независимая русская оценка ещё не выполнены.')
para('Границы текущего решения: статические изображения; нет контекста переписки и персонализации. В каталоге есть сторонние персонажи и синтетические открытки. Успешное онлайн-демо остаётся обязательным условием итоговой оценки.','SmallRU')

head(2,'Источники и структура датасета')
table(['Часть','Получено после очистки','Разметка / источник'],[
 ('Telegram stickers','30 000','nyuuzyou/stickers [1]; исходная метка - emoji, описаний предложениями нет'),
 ('StickerQueries',str(stats['sources']['StickerQueries']),'Английские поисковые запросы, проверенные авторами набора [2]'),
 ('Synth-GCD','100','Синтетические открытки с английскими надписями внутри изображения [3]'),
 ('OpenMoji-карточки','240','12 поводов × 2 рисунка × 5 шаблонов × с текстом/без текста [4]'),
],[4,3.4,9.6])
para('Для большого каталога загружен valid.zip объёмом 6 598 157 862 байта; проверен опубликованный SHA-256. Среди кандидатов обеспечено начальное покрытие доступных emoji-категорий, затем выбран перемешанный остаток, seed=42. Этот архив используется как сырьё для собственного корпуса: официальное validation-разбиение источника не является нашим benchmark.')
para(f'Английский CSV StickerQueries содержит 1 115 строк: 578 PNG и 537 GIF/WEBM. Все 578 PNG загружены; {source_stats["exact_duplicate_count"]} повторяющихся изображений объединены. После очистки сохранено {source_stats["source_query_count"]:,} пар «поисковая фраза - изображение». Запятые трактуются как разделители списка по схеме источника. Фразы не равнозначны независимым пользовательским сессиям.')
para('Лицензии и происхождение','SubRU')
para('Авторы карточек датасетов заявляют WTFPL для [1], MIT для [2], Apache-2.0 для [3]. Графика OpenMoji [4] - CC BY-SA 4.0; производные композиции сохраняют атрибуцию и лицензию. Декларация набора не является независимой проверкой прав на каждый исходный рисунок. Корпус автоматически публично не публикуется.')
table(['Файл / поля','Назначение'],[
 ('manifest.jsonl','item_id, путь, источник, ревизия, контрольные суммы, размеры, прозрачность, группа и split'),
 ('labels/queries_en.jsonl','query_id, текст, язык, известные позитивы, split; не входит в признаки поиска'),
 ('annotations.jsonl (план)','VLM-caption, эмоция, стиль, OCR, синтетические train-запросы; происхождение разметки'),
 ('audit/','Ошибки загрузки, точные дубликаты, кандидаты на близость и статистики'),
],[6,11])

head(3,'EDA: измеренные свойства и баланс')
figure(ROOT/'evidence/catalog/eda.png',17)
para('Рис. 1. Фактический состав каталога, split, доля непрозрачности и размеры групп после исправления правила группировки. Логарифмическая ось используется только для числа групп.','SmallRU')
table(['Проверка','Результат','Следствие'],[
 ('Состав',f'{stats["kinds"]["sticker"]:,} стикера; {stats["kinds"]["postcard"]} открыток','Открытки составляют около 1,1%; общая метрика может скрыть их качество'),
 ('Прозрачность',f'{stats["transparent_images"]:,} изображений с alpha <255','Для энкодера - композиция на белом фоне, без потери оригинального alpha'),
 ('Точные дубликаты',f'{source_stats["exact_duplicate_count"]} объединены; {stats["unique_pixel_hashes"]:,} уникальных хешей','Одинаковые пиксели не попадают в разные группы'),
 ('Декодирование и загрузка',f'Включённые файлы проверены; ошибок загрузки {source_stats["failed_count"]}','Числа относятся к текущему snapshot, а не ко всему исходному набору'),
],[4.2,5.1,7.7])
para('Неизвестные сейчас свойства: доля корректных будущих caption/OCR, качество русского перевода, частоты настоящих пользовательских сценариев. Их нельзя вывести из размеров картинок или наличия alpha.','SmallRU')

head(4,'EDA: примеры и пробелы покрытия')
# Four deterministic examples per source, directly from the current manifest.
selected=[]
for source in stats['sources']:
    selected += [r for r in items if r['source']==source][:4]
canvas=PILImage.new('RGB',(800,840),'#f5f7fb');draw=ImageDraw.Draw(canvas)
for i,r in enumerate(selected):
    with PILImage.open(ROOT/'data/catalog'/r['image_path']) as im:
        im=im.convert('RGBA');im=PILImage.alpha_composite(PILImage.new('RGBA',im.size,'white'),im).convert('RGB')
        im=ImageOps.contain(im,(180,175))
    x,y=(i%4)*200,(i//4)*210;canvas.paste(im,(x+(200-im.width)//2,y+5+(175-im.height)//2))
    draw.text((x+8,y+187),r['source'][:26],fill='#27334b')
preview=ROOT/'evidence/catalog/report_examples.jpg';canvas.save(preview,quality=92)
figure(preview,13.1)
para('Рис. 2. По четыре примера каждого источника; первые ID в сохранённом manifest. Прозрачность показана на белом фоне. Изображения не выбирались по успешности поиска.','SmallRU')
para('Корпус стикеров существенно больше корпуса открыток. Открытки полностью синтетические: 100 сгенерированных изображений и 240 шаблонных композиций. Для последних имеется лишь 24 независимых мотива. Повторение рамок и цветов не заменяет разнообразие сюжетов и реальных поздравительных формулировок.')
para('GIF/WEBM исключены: смысл движения может потеряться при выборе одного кадра. У большого Telegram-корпуса вместо описания только emoji; одинаковая метка может объединять очень разные эмоции и сцены. Поэтому перенос метки emoji в длинное описание не считается качественной разметкой.')

head(5,'Представления и архитектура')
table(['Компонент','Представление','Обоснование / ограничение'],[
 ('Изображение','CLIP ViT-B/32, 512D, L2','Готовая связь изображения с текстом; ограничен на эмоциях и тонком смысле мемов'),
 ('Запрос RU/EN','Согласованный multilingual CLIP text encoder [5], 512D','Русский и английский находятся в пространстве выбранного image encoder'),
 ('Визуальное описание','Qwen2.5-VL-7B [6] → русский caption → text encoder','Добавляет явно выраженные предметы, стиль и эмоции; возможны галлюцинации'),
 ('Надпись','Отдельное поле OCR → text encoder; char TF-IDF baseline','Точная надпись может менять смысл; OCR необходимо проверять отдельно'),
 ('Метаданные','Тип и источник; группа для split','Фильтр стикеров/открыток. Оценочные запросы и эталонные сценарии не индексируются'),
],[3.8,5.7,7.5])
para('Визуальные и текстовые векторы согласованы: используются image encoder исходного CLIP и специально дистиллированный в его пространство текстовый encoder. Произвольный BERT нельзя подставить в тот же скалярный продукт без выравнивания. Для прозрачных файлов фон явно приводится к белому; оригинал сохраняется.')
para('Поиск и обучение','SubRU')
para('Baseline вычисляет cosine(q, image). Гибрид смешивает image, caption и OCR с начальными весами 0,65 / 0,25 / 0,10, перенормированными по доступным ветвям. При пустом тексте работает визуальная ветвь. Обучаемые части: две residual-проекции rank=64 для query/image и три веса смешивания; основные энкодеры заморожены.')
para('Целевая функция - multi-positive InfoNCE, temperature=0,07. Известные позитивы одинакового запроса не становятся отрицательными друг для друга; близкие изображения одной группы маскируются среди негативов. Неразмеченные семантические аналоги всё же могут остаться ложными отрицательными: это ограничение исходных меток.')
para('Для 30 тыс. векторов 512D один FP32-массив занимает примерно 61 MB. Используется точное матричное произведение; сложный ANN-индекс пока не обоснован измерениями. DINO требует дополнительного выравнивания с текстом. SigLIP2 - кандидат следующего сравнения, но ещё не измеренная альтернатива.')

head(6,'Разбиение, перевод и протокол оценки')
splits=stats['splits']
para(f'Полный каталог разделён на {splits.get("train",0):,} train, {splits.get("dev",0):,} dev и {splits.get("test",0):,} test изображений. Сначала объединяются точные дубликаты и известные семьи рисунков, затем применяется эвристика dHash≤3 с близкими пропорциями и средним цветом. Хеш группы с seed=42 задаёт split. Всего {stats["num_groups"]:,} групп.')
para(f'При аудите обнаружены ложные объединения: одинаковые рамки открыток и большие пустые поля стикеров. Исправления: разные известные семейства рисунков не сливаются только по dHash; остальные кандидаты проверяются на обрезанных по содержимому миниатюрах 64×64 (IoU масок ≥0,85; средняя RGB-ошибка ≤0,03). Самая большая группа теперь содержит {stats["largest_group"]} изображений. Pack ID исходного Telegram-корпуса отсутствуют; независимость настоящих паков не гарантирована.')
para('Защита от утечки','SubRU')
para('Английская фраза и русский перевод остаются в split исходного изображения. VLM-caption строится только по пикселям и не получает человеческих поисковых запросов. Синтетические ситуации использования применяются только как train-метки; не включаются напрямую в поисковый индекс. Изображения теста присутствуют среди кандидатов, но их пары с запросами не обучают модель.')
table(['Метрика','Смысл'],[
 ('Hit@5','Есть ли известный релевантный объект в первых пяти позициях'),
 ('Recall@k по известным позитивам','Доля найденных известных позитивов; не полный recall при неполной разметке'),
 ('NDCG@5','Дисконтирование низких позиций; для будущих оценок 0/1/2 учитывает степень релевантности'),
 ('MRR@10','Обратная позиция первого известного позитива в top-10'),
 ('p50/p95 и память','Раздельно online-поиск, кодирование запроса и offline-индексация'),
],[5.4,11.6])
para('В исходных query-image парах не перечислены все подходящие картинки. В вычисленной proxy-оценке неразмеченные кандидаты трактуются как 0; это позволяет проверить нахождение известного позитива, но не доказывает общую семантическую релевантность. Для итогового отчёта нужны независимые русские запросы и слепая оценка общего пула выдачи нескольких методов.')
para('План: 100–200 русских запросов, стикеры и открытки, визуальные и текстовые ограничения, no-match случаи; оценки 0/1/2. На части данных - второй оценщик. Машинный перевод и синтетические запросы отдельно маркируются. Интервалы неопределённости считаются по группам изображений, а не по коррелированным перефразировкам.','SmallRU')

head(7,'Проверка подхода на отдельном CPU-пилоте')
para(f'Пилот содержит 232 стикера StickerQueries и 240 OpenMoji-карточек. Это отдельный ранний snapshot с собственным split: {config["training_queries"]} обучающих, {config["dev_queries"]} validation и {metrics["methods"]["image"]["query_count"]} тестовых английских запросов. Тест относится к {metrics["methods"]["image"]["group_count"]} группам стикеров; качество открыток и русских запросов этими числами не измеряется.')
m=metrics['methods']['image']['metrics']
table(['Метод','Hit@5','NDCG@5','MRR@10'],[
 ('Замороженный multilingual CLIP',f'{m["Hit@5"]["mean"]:.4f}',f'{m["NDCG@5"]["mean"]:.4f}',f'{m["MRR@10"]["mean"]:.4f}'),
 ('Выбранный по dev checkpoint','То же','То же','То же'),
],[7.7,3.1,3.1,3.1])
figure(ROOT/'evidence/cpu_pilot/training.png',17)
para('Рис. 3. Пять эпох обучения на замороженных признаках. Train-loss уменьшается, validation остаётся ниже исходного baseline. Выбор checkpoint включает эпоху 0.','SmallRU')
para(f'Исходный dev NDCG@5 = {training["initial_dev_ndcg5"]:.4f}; лучший checkpoint - эпоха 0. Поэтому совпадение test-метрик двух строк объясняется возвратом исходной модели, а не улучшением от обучения. Пять эпох: AdamW, lr=0,001, batch=64, rank=64. Повторный CPU-запуск с seed=42 дал идентичный журнал потерь и dev-метрик.')
para('Полученный результат не опровергает дообучение на полном корпусе: здесь мало размеченных изображений, отсутствуют новые caption/OCR-признаки и русская supervision. Одновременно он не даёт оснований обещать улучшение. На этапе 2 сравнение проводится на одном полном каталоге с заранее выбранным dev-протоколом.')
para('Проверены 9 автоматических тестов: метрики, градиенты multi-positive loss, прозрачность, группировка и перенос архивов. Streamlit-интерфейс принимает русский запрос и возвращает выдачу без исключений. GPU-аннотация, генерация и реальное подключение YT пока не проверены.','SmallRU')

head(8,'Гипотезы и дальнейшая работа')
table(['Шаг','Эксперимент / действие','Критерий завершения'],[
 ('1. Разметка','Qwen2.5-VL на 500 примерах; проверить 50–100 вручную','Ошибки OCR, caption и эмоций посчитаны; измерена скорость'),
 ('2. Масштабирование','Полная разметка; EN→RU перевод исходных запросов','Прошли schema-checks; сохранены модель, ревизия, prompt и источники меток'),
 ('3. Сравнение','OCR TF-IDF; image-only; гибрид; обученные adapters','Один каталог, один test; выбор модели только по dev'),
 ('4. Русская оценка','Независимые RU-запросы, pooled-оценки 0/1/2','Метрики по стикерам/открыткам, надписям и no-match; разбор ошибок'),
 ('5. Генерация','FLUX.1-schnell + U2Net/rembg','Минимум пять реальных запусков, исходники/маски/PNG и время'),
 ('6. Сдача','README, репозиторий, PDF 10–15 страниц, онлайн-демо','Работающее решение и объяснение ограничений; демо до 15 минут'),
],[2.8,7,7.2])
para('Проверяемые гипотезы','SubRU')
para('H1: OCR улучшает запросы по надписи, но мало помогает на стикерах без текста. H2: визуальные caption и эмоции помогают запросам о намерении по сравнению с image-only. H3: обучение проекций на human-source и слабой разметке улучшает held-out retrieval. H4: перевод supervision помогает русским запросам; проверка требует отдельных русских меток.')
para('Для генерации используется готовая FLUX.1-schnell [7], затем U2Net через rembg [8]. Исходник и RGBA сохраняются отдельно. Это реализация обязательного функционала этапа 2; код подготовлен, но результат генерации в текущем отчёте не предъявляется как выполненный. Точная кириллица не обещается: первая версия генерирует без надписей.')
para('Рабочая среда пользователя: 4 × A100 80 GB. Предполагается предварительная загрузка данных и моделей, размещение архивов в YT и скачивание на локальный диск GPU-узла перед работой. Подготовлены checksum-архивы по 500 картинок и скрипты переноса. Конкретный кластер, каталог и рецепт запуска ещё не заданы.')
para('Категории будущих ошибок: неверная эмоция при верном объекте; OCR и отрицание; ирония; культурная отсылка; потеря движения; открытка другого повода; отсутствие нужного сюжета; релевантный, но неразмеченный альтернативный ответ. Отдельно анализируется качество маски при удалении фона.','SmallRU')

head(9,'Воспроизводимость и источники')
para('Комплект sticker_search_project.zip содержит код, manifest полного каталога, README, готовый CPU-пилот, его эмбеддинги, журналы экспериментов и checkpoint. Полные 30 тысяч исходных PNG восстанавливаются из закреплённого архива с проверкой каждого файла. Внешний GitHub/GitLab-репозиторий ещё не опубликован; его ссылку нужно добавить при сдаче.')
para('Основные команды','SubRU')
for text in ['python -m sticker_search.restore_catalog --output data/catalog',
             'python -m sticker_search.audit --catalog data/catalog',
             'bash scripts/run_gpu.sh',
             'python -m streamlit run app.py']:
    para(escape(text),'SmallRU')
para('Прямые зависимости и модели закреплены в pyproject.toml и configs/model_revisions.json. Фактическая CPU-среда записана отдельно. Полный lock GPU-среды, версия CUDA и измерения A100 будут добавлены после запуска. Код и текст подготовлены с помощью AI; численные результаты получены выполнением скриптов. Ручная независимая оценка не заявляется.')
refs=[
 ('[1] Telegram stickers, источник PNG','https://huggingface.co/datasets/nyuuzyou/stickers'),
 ('[2] StickerQueries, человеческие query-image пары','https://huggingface.co/datasets/metchee/sticker-queries'),
 ('[3] Synthetic Greeting Cards','https://huggingface.co/datasets/gauravs101/synthetic-greeting-cards'),
 ('[4] OpenMoji: графика и лицензия','https://github.com/hfg-gmuend/openmoji'),
 ('[5] Согласованный мультиязычный CLIP','https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1'),
 ('[6] Qwen2.5-VL-7B-Instruct','https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct'),
 ('[7] FLUX.1-schnell','https://huggingface.co/black-forest-labs/FLUX.1-schnell'),
 ('[8] rembg, удаление фона','https://github.com/danielgatis/rembg'),
 ('[9] YTsaurus Python API','https://ytsaurus.tech/docs/en/api/python/userdoc'),
]
for name,url in refs:para(f'<b>{escape(name)}</b><br/>{escape(url)}','SmallRU')
para('Материалы курса: лекции 5 «Metric learning», 6 «CLIP, DINO», 9 «Распознавание текста», 12 «Мультимодальные модели». Дата проверки внешних источников: 27.09.2026.','SmallRU')


def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#dce2ef'));canvas.line(2*cm,1.55*cm,19*cm,1.55*cm)
    canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#69778d'))
    canvas.drawString(2*cm,1.05*cm,'Подбор стикеров и открыток · промежуточный отчёт')
    canvas.drawRightString(19*cm,1.05*cm,str(doc.page))


target=ROOT/'output/stage1_report.pdf';target.parent.mkdir(parents=True,exist_ok=True)
doc=SimpleDocTemplate(str(target),pagesize=(21*cm,29.7*cm),leftMargin=2*cm,rightMargin=2*cm,topMargin=1.7*cm,bottomMargin=2*cm,
                     title='Интеллектуальный подбор стикеров и открыток. Этап 1',author='Учебный проект')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(target)
