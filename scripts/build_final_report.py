"""Build the final report from recorded pod outputs, without rerunning models.

Run: python scripts/build_final_report.py
Dependencies: reportlab, numpy, matplotlib, Pillow (no CUDA or torch required).
"""
from pathlib import Path
from collections import Counter
from html import escape, unescape
import base64, io, json, re, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf'; OUT.mkdir(parents=True,exist_ok=True)
ASSETS=ROOT/'evidence/final_v1/report_assets'; ASSETS.mkdir(parents=True,exist_ok=True)
FINAL=zipfile.ZipFile(ROOT/'evidence/final_v1/final_results.zip')
DIAG=zipfile.ZipFile(ROOT/'evidence/full_v2/retrieval_diagnostics.zip')
V3=zipfile.ZipFile(ROOT/'evidence/full_v2/retrieval_v3_results.zip')
def arc_json(z,suffix):
    names=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
    if len(names)!=1: raise ValueError((suffix,names))
    return json.loads(z.read(names[0]))
def arc_lines(z,suffix):
    names=[n for n in z.namelist() if n==suffix or n.endswith('/'+suffix)]
    if len(names)!=1: raise ValueError((suffix,names))
    return [json.loads(x) for x in z.read(names[0]).decode().splitlines()]
STATS=json.loads((ROOT/'evidence/catalog/summary.json').read_text())
TRAIN=json.loads((ROOT/'evidence/full_v2/completed_training_20260928.json').read_text())
TEST={l:arc_json(FINAL,f'test_{l}/metrics.json') for l in ['en','ru']}
DEV={l:arc_json(DIAG,f'dev_{l}/metrics.json') for l in ['en','ru']}
CAL={l:arc_json(V3,f'dev_{l}/metrics.json') for l in ['en','ru']}
ENV=arc_json(FINAL,'environment.json')
assert arc_json(FINAL,'status.json')['errors']==[]
assert arc_json(FINAL,'frozen_selection.json')['selected_epoch']==4
assert all(TEST[l]['catalog_size']==30914 for l in TEST)

def metric(data,method,name='NDCG@5'):
    return data['methods'][method]['metrics'][name]['mean']

# Recompute metrics from per-query outputs. This does not select or tune a model.
analysis={'bootstrap':{'seed':42,'replicates':10000,'unit':'image group','paired':True},'comparisons':{}}
for lang in ['en','ru']:
    rows=arc_lines(FINAL,f'test_{lang}/cases.jsonl')
    by={m:{x['query_id']:x for x in rows if x['method']==m} for m in ['image','rrf','learned']}
    groups=sorted({x['group_id'] for x in by['image'].values()})
    assert len(groups)==86
    for m in by:
        assert len(by[m])==680 and set(by[m])==set(by['image'])
        assert abs(np.mean([x['metrics']['NDCG@5'] for x in by[m].values()])-metric(TEST[lang],m))<1e-12
    rng=np.random.default_rng(42)
    for m in ['rrf','learned']:
        diffs={g:[] for g in groups}
        for q,x in by[m].items():
            assert x['group_id']==by['image'][q]['group_id']
            diffs[x['group_id']].append(x['metrics']['NDCG@5']-by['image'][q]['metrics']['NDCG@5'])
        sums=np.array([sum(diffs[g]) for g in groups]); ns=np.array([len(diffs[g]) for g in groups])
        boots=[]
        for _ in range(10000):
            idx=rng.integers(0,len(groups),len(groups));boots.append(sums[idx].sum()/ns[idx].sum())
        analysis['comparisons'][f'{lang}_{m}_minus_image']={'mean':float(sums.sum()/ns.sum()),'ci95':np.quantile(boots,[.025,.975]).tolist()}
reviews=arc_lines(FINAL,'review/search_examples.jsonl')
analysis['warm_latency']={m:{k:{'p50':float(np.median([x[k] for x in reviews if x['method']==m])),'p95':float(np.quantile([x[k] for x in reviews if x['method']==m],.95))} for k in ['encoder_ms','search_ms','warm_end_to_end_ms']} for m in ['image','rrf','learned']}
gen_names=sorted(n for n in FINAL.namelist() if n.endswith('/generation.json'))
GEN=[json.loads(FINAL.read(n)) for n in gen_names]
assert len(GEN)==5
analysis['generation_seconds']=[g['seconds'] for g in GEN]
analysis['visual_review']={'reviewer':'AI assistant, not an independent human assessor','search_probes':18,'generation_pairs':5,'findings':['laughter confused with crying','birthday confused with congratulations','learned top-1 good-morning matches but top-3 includes good-night','robot cutout loses opaque body and thin details','space setting absent in generated robot example']}
(ROOT/'evidence/final_v1/analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':170})
blue='#3554B5'; teal='#148478'; orange='#D98A30'; ink='#1D2D49'
fig,ax=plt.subplots(1,2,figsize=(10,3.0))
names=['Telegram','StickerQueries','OpenMoji','Synth-GCD']; counts=[30000,574,240,100]
ax[0].barh(names[::-1],counts[::-1],color=[orange,teal,blue,'#839BD1']);ax[0].set_xscale('log');ax[0].set_xlabel('Изображений, логарифмическая шкала')
for i,n in enumerate(counts[::-1]):ax[0].text(n*1.06,i,str(n),va='center',fontsize=9)
ax[0].set_xlim(50,80000)
ax[1].bar(['train','dev','test'],[21805,4469,4640],color=[blue,teal,orange]);ax[1].set_ylabel('Изображений')
for i,n in enumerate([21805,4469,4640]):ax[1].text(i,n+450,f'{n:,}'.replace(',',' '),ha='center',fontsize=10)
ax[1].set_ylim(0,24500);fig.tight_layout();fig.savefig(ASSETS/'eda.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,3.2));h=TRAIN['history_epochs_1_to_10'];ep=[x['epoch'] for x in h]
ax[0].plot(ep,[x['loss'] for x in h],'-o',color=blue);ax[0].set(xlabel='Эпоха',ylabel='Train loss',xticks=[1,2,4,6,8,10])
ax[1].plot([0]+ep,[.0019989116431524613]+[x['dev_ndcg5'] for x in h],'-o',color=teal);ax[1].axvline(4,ls='--',color=orange,label='Выбранная эпоха 4');ax[1].set(xlabel='Эпоха',ylabel='Dev NDCG@5, EN+RU',xticks=[0,2,4,6,8,10]);ax[1].legend(fontsize=8)
fig.tight_layout();fig.savefig(ASSETS/'training.png');plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(10,3.0),sharey=True)
for a,l in zip(ax,['en','ru']):
    for i,(m,c) in enumerate(zip(['image','rrf','learned'],[blue,teal,orange])):
        stat=TEST[l]['methods'][m]['metrics']['NDCG@5']; v=stat['mean'];low,high=stat['ci95']
        a.errorbar(i,v,yerr=[[v-low],[high-v]],fmt='o',color=c,capsize=5,markersize=7)
    a.set(xticks=range(3),xticklabels=['Image','RRF','Learned'],title=l.upper()+' test, 680 запросов');a.grid(axis='y',alpha=.2)
ax[0].set_ylabel('NDCG@5 и 95% bootstrap CI');fig.tight_layout();fig.savefig(ASSETS/'test_ci.png');plt.close(fig)

# Reuse real thumbnail pixels embedded in the supplied HTML, never generated substitutes.
page=FINAL.read('review/search_review.html').decode()
sections=re.findall(r'<section>(.*?)</section>',page,re.S)
CACHE={};QUERIES=[]
for section in sections:
    query=unescape(re.search(r'<h2>(.*?)</h2>',section).group(1));QUERIES.append(query)
    for article in re.findall(r'<article>(.*?)</article>',section,re.S):
        key=re.search(r'alt="([^"]+)',article).group(1)
        im=PILImage.open(io.BytesIO(base64.b64decode(re.search(r'src="data:image/png;base64,([^"]+)',article).group(1)))).convert('RGB')
        path=ASSETS/(key+'.png');im.save(path);CACHE[key]=path

# Five original/cutout pairs. Dark backgrounds expose mask damage.
canvas=PILImage.new('RGB',(1250,565),'white');d=ImageDraw.Draw(canvas)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
for i,(n,g) in enumerate(zip(gen_names,GEN)):
    folder=n.rsplit('/',1)[0]
    d.text((i*250+12,5),f'{i+1}. seed={g["seed"]}',font=font,fill=ink)
    orig=PILImage.open(io.BytesIO(FINAL.read(folder+'/original.png'))).convert('RGBA')
    cut=PILImage.open(io.BytesIO(FINAL.read(folder+'/sticker.png'))).convert('RGBA')
    dark=PILImage.alpha_composite(PILImage.new('RGBA',cut.size,'#243046'),cut)
    for j,im in enumerate([orig,dark]):
        thumb=ImageOps.contain(im,(240,240)).convert('RGB');canvas.paste(thumb,(i*250+5,32+j*260))
canvas.save(ASSETS/'generation_pairs.png')

font_dir=Path('/usr/share/fonts/truetype/dejavu')
for name,filename in [('DV','DejaVuSans.ttf'),('DV-B','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(font_dir/filename)))
pdfmetrics.registerFontFamily('DV',normal='DV',bold='DV-B',italic='DV',boldItalic='DV-B')
styles={
 'body':ParagraphStyle('body',fontName='DV',fontSize=9.6,leading=14,spaceAfter=9,textColor=colors.HexColor(ink)),
 'small':ParagraphStyle('small',fontName='DV',fontSize=8.0,leading=11.3,spaceAfter=7,textColor=colors.HexColor('#536279')),
 'head':ParagraphStyle('head',fontName='DV-B',fontSize=19,leading=24,spaceAfter=14,textColor=colors.HexColor(ink)),
 'title':ParagraphStyle('title',fontName='DV-B',fontSize=25,leading=31,spaceAfter=18,textColor=colors.HexColor(ink)),
 'sub':ParagraphStyle('sub',fontName='DV-B',fontSize=11,leading=15,spaceAfter=8,spaceBefore=7,textColor=colors.HexColor(blue)),
 'cell':ParagraphStyle('cell',fontName='DV',fontSize=8.4,leading=11.5,textColor=colors.HexColor(ink)),
 'tiny':ParagraphStyle('tiny',fontName='DV',fontSize=7.2,leading=10,textColor=colors.HexColor(ink)),
}
story=[]
def p(text,style='body'):story.append(Paragraph(text,styles[style]))
def head(n,title):
    if story:story.append(PageBreak())
    p(f'ИТОГОВЫЙ ПРОЕКТ / {n:02d}', 'small');p(title,'head')
def table(headers,rows,widths=None,small=False):
    sty=styles['tiny' if small else 'cell']
    data=[[Paragraph('<b>'+escape(str(x))+'</b>',sty) for x in headers]]
    data += [[Paragraph(escape(str(x)),sty) if not hasattr(x,'wrap') else x for x in row] for row in rows]
    t=Table(data,colWidths=[w*cm for w in widths] if widths else None,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF9')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]));story.append(t);story.append(Spacer(1,9))
def fig(path,w=17):
    with PILImage.open(path) as im:iw,ih=im.size
    story.append(Image(str(path),width=w*cm,height=w*cm*ih/iw,hAlign='LEFT'));story.append(Spacer(1,6))
def num(x):return f'{x:.4f}'

head(1,'Результат и постановка задачи')
p('Интеллектуальная система<br/>подбора стикеров и открыток','title')
p('Этап 2. Итоговый отчёт · Р. Гулямов · 28 сентября 2026','small')
p('Реализован воспроизводимый прототип: каталог из <b>30 914 изображений</b>, мультиязычный поиск, визуальные описания и OCR, обучение адаптеров, оценка на dev/test и генерация стикера с удалением фона. Все численные результаты ниже получены из сохранённых запусков на рабочем GPU-узле; качественный разбор выполнен с помощью AI и не заменяет независимую оценку людьми.')
p('<b>Главный вывод:</b> обучение выполнилось корректно, но устойчивое улучшение русского поиска не подтверждено. Выбранный до теста адаптер получил RU test NDCG@5 = 0,0080 против 0,0144 у image-only. На EN знак разницы противоположный. Доверительные интервалы разностей включают ноль. Проект демонстрирует полный процесс и ограничения, а не готовое промышленное качество.')
table(['Требование','Что реализовано / проверено'],[
 ('Свободный запрос RU/EN','Общее пространство текста и изображения; ранжирование и фильтр типа'),
 ('Визуальная семантика и надписи','CLIP + отдельные caption/OCR-признаки; baseline и ablation'),
 ('Поиск по базе','Точный поиск по 30 914 кандидатам; top-12 в Streamlit'),
 ('Новый стикер при плохой выдаче','Кнопка генерации; FLUX + U2Net; пять реальных примеров'),
 ('Воспроизводимость','Ревизии моделей, seed, хеши индекса и checkpoint; отдельные скрипты'),
],[5.3,11.7])
p('Границы: статические картинки, без контекста переписки и персонализации. Необходимы точное намерение, тон и повод, а не только совпадение объекта. Автоматический порог «ничего подходящего» не обучен: решение о генерации принимает пользователь. Онлайн-защита и публикация ссылки на репозиторий остаются организационными шагами.','small')

head(2,'Датасет: происхождение и структура')
table(['Источник','Объём','Разметка и условия'],[
 ('nyuuzyou/stickers [1]','30 000','Статичные Telegram-стикеры, исходные emoji; в карточке набора указана WTFPL'),
 ('StickerQueries [2]','574','Человеческие английские query-image пары; в карточке указана MIT'),
 ('Synth-GCD [3]','100','Синтетические открытки с английской надписью внутри изображения; Apache-2.0'),
 ('OpenMoji-композиции [4]','240','12 поводов × 2 мотива × 5 шаблонов × наличие текста; CC BY-SA 4.0'),
],[4.6,1.8,10.6])
p('Большой корпус построен из valid.zip nyuuzyou размером 6 598 157 862 байта с проверкой SHA-256. Выбраны 30 тысяч изображений с начальным покрытием emoji-категорий и перемешиванием остатка (seed=42). Исходный valid-раздел используется как сырьё; наш split и метрики не являются официальным benchmark этого набора.')
p('Из английского CSV StickerQueries взяты все 578 PNG; четыре точных повтора объединены. 537 GIF/WEBM исключены, чтобы не терять смысл движения при сведении к одному кадру. После обработки получена 4 171 поисковая фраза. Поле labeled_queries разбирается как список по схеме источника [2].')
table(['Артефакт','Содержимое'],[
 ('manifest.jsonl','item_id, image_path, source, revision, checksum, width/height, group_id, split'),
 ('labels/queries_en.jsonl','Идентификатор запроса, текст, известные позитивы и split'),
 ('captions/annotations.jsonl','Русское описание, эмоция, стиль, OCR и происхождение VLM-ответа'),
 ('index/items.jsonl + vectors.npz','Порядок объектов, метаданные; image/caption/OCR по 512 измерений'),
 ('queries_*/queries.npy','Векторы запроса в порядке соответствующего JSONL'),
],[6.1,10.9])
p('Декларации лицензий взяты из карточек источников; они не являются независимой проверкой прав на каждого стороннего персонажа. Атрибуция OpenMoji сохранена в evidence/full_catalog/licenses. Код, метаданные и выбранные примеры отделены от многогигабайтного корпуса и весов.','small')

head(3,'EDA и контроль качества данных')
fig(ASSETS/'eda.png')
p('Рис. 1. Реальный состав корпуса и собственное групповое разбиение. Логарифмическая шкала слева показывает сильный дисбаланс источников.','small')
table(['Проверка','Измерение','Следствие'],[
 ('Тип контента','30 574 стикера; 340 открыток (1,10%)','Качество открыток нельзя вывести из общей sticker-метрики'),
 ('Прозрачность','23 905 изображений (77,33%)','Перед CLIP накладываются на белый фон; оригинал не заменяется'),
 ('Уникальность','30 914 уникальных pixel SHA-256; 4 повтора объединены','Идентичные пиксели не разделяются между split'),
 ('Группы','30 508; максимальный размер 10','Варианты одного рисунка связаны при разбиении'),
 ('Файлы','Включённые изображения декодированы; ошибок загрузки 0','Это проверка доступности и формата, не семантического качества'),
],[3.5,5.6,7.9])
p('Для поиска близких изображений использованы dHash≤3, пропорции и средний цвет. Ранний вариант ошибочно склеивал рамки и пустые поля. Добавлена проверка обрезанных по содержимому миниатюр: IoU масок ≥0,85 и RGB MAE≤0,03; известные разные семейства открыток не объединяются только по dHash.')
p('Split задаётся хешем группы с seed=42: train 21 805, dev 4 469, test 4 640 изображений. Pack ID у основного Telegram-корпуса недоступны, поэтому разделение истинных паков не гарантировано. OpenMoji-часть имеет 24 независимых мотива: 240 шаблонных вариантов не равны 240 независимым сюжетам.')
p('Пробелы покрытия: мало открыток, все они синтетические; отсутствует анимация и независимая русская разметка намерений. Увеличение числа неразмеченных стикеров само по себе расширяет каталог, но не добавляет обучающих query-image пар.','small')

head(4,'Аннотации, перевод и происхождение меток')
p('Qwen2.5-VL-7B-Instruct [6] получает изображение без исходного поискового запроса и возвращает JSON. Для текстового признака объединяются caption_ru, emotion_ru и style_ru; распознанная надпись хранится отдельно. Аннотация распределена по четырём GPU, по независимому процессу и копии модели на устройство.')
table(['Шаг','Фактический результат','Правило обработки'],[
 ('VLM caption/OCR','30 875 из 30 914 (99,87%)','39 объектов остаются с image-признаками; покрытие отмечено явно'),
 ('Непустая OCR-ветвь','10 958 из 30 914 (35,45%)','Пустой OCR даёт нулевой вектор; это не оценка точности OCR'),
 ('Перевод EN→RU','4 124 валидных ответа + 47 исправлений','47 ответов добавлены ассистентом на CPU; происхождение сохранено'),
 ('Итоговые запросы','4 171 EN + 4 171 RU = 8 342','RU - производные машинные метки, не независимая gold-разметка'),
],[3.4,6.0,7.6])
p('Ошибки формата сохраняются вместе с исходным ответом. Повторный запуск обрабатывает только пропуски; успешные записи и хеш конфигурации не меняются. Это позволило продолжить после обрыва JSON и циклического повторения символов, не повторяя всю разметку.')
p('Корректный JSON не гарантирует правильный смысл','sub')
table(['Исходная фраза','Обнаруженный перевод','Проблема'],[
 ('im a happy hippo','я счастливый хиппи','Подмена объекта'),
 ('arrest me','Заткнись!','Полная смена намерения'),
 ('shocked face','восхищенный вид','Подмена эмоции'),
],[5.1,5.1,6.8])
p('Эти ошибки обнаружены при диагностике dev и оставлены в зафиксированной версии benchmark; выборочно исправлять только неудачные оценочные запросы нельзя. Следующая версия требует аудита всех split и новых независимых RU-запросов. 47 исправлений парсинга также не считаются независимой человеческой проверкой.')
p('Ранний аудит показал галлюцинации VLM в сценариях использования. Поэтому usage_queries_ru исключены из обучения full_v2; caption и OCR оставлены только как слабые признаки. Точность OCR по CER/WER не измерена: нет эталонных транскрипций.','small')

head(5,'Представления и архитектура поиска')
table(['Вход / компонент','Представление','Почему выбран'],[
 ('Изображение','CLIP ViT-B/32, L2, 512D','Готовое визуально-текстовое пространство [5, 9]'),
 ('Запрос RU/EN','Multilingual CLIP text, L2, 512D','Текстовый student дистиллирован в пространство того же image encoder [5]'),
 ('Caption / OCR','Тот же text encoder, по 512D','Явно учитывает описание, эмоцию, стиль и надпись'),
 ('Query / image adapter','Residual MLP 512→64→512','Малая обучаемая часть при ограниченной supervision'),
 ('Хранение','JSONL + NumPy FP32; item_id','Простой проверяемый порядок строк и соответствие метаданным'),
],[3.7,6.0,7.3])
p('<b>Offline:</b> картинки → очистка/группы → VLM и CLIP → массивы признаков; train-пары → обучение адаптеров. <b>Online:</b> запрос → text encoder → выбранный scoring → фильтр типа → top-k → изображения. Генерация - отдельная ветка по действию пользователя.')
p('Сравниваемые методы','sub')
p('<b>Image:</b> скалярное произведение L2-векторов запроса и изображения (cosine). <b>Lexical:</b> символьный TF-IDF по OCR. <b>Hybrid:</b> сумма image/caption/OCR с весами 0,65/0,25/0,10. Для отсутствующей ветви вес зануляется и остальные перенормируются; итоговая сумма повторно не нормируется.')
p('<b>Learned:</b> A(x)=normalize(x+W<sub>up</sub> GELU(W<sub>down</sub>x)); отдельные адаптеры для query/image, обучаемые softmax-веса ветвей. Caption/OCR-энкодеры заморожены. Верхние матрицы адаптеров изначально нулевые: эпоха 0 совпадает с hybrid. Всего 131 075 обучаемых параметров.')
p('<b>RRF:</b> 0,8/(60+rank<sub>image</sub>) + 0,2/(60+rank<sub>caption</sub>); первая позиция имеет rank=1. Отсутствующие признаки не вносят вклад. Ранги помогают уменьшить влияние разных диапазонов сырых score; они не исправляют ошибочный caption.')
p('DINO без дополнительного выравнивания не даёт готового поиска по тексту. Более крупный CLIP/SigLIP и reranker оставлены для следующего опыта. Полное дообучение backbone не выбрано из-за малого числа размеченных изображений и риска переобучения.','small')

head(6,'Обучение и выбор checkpoint')
fig(ASSETS/'training.png')
p('Рис. 2. Train-loss убывает, но dev-качество после четвёртой эпохи не улучшается. Точка 0 - исходный hybrid, а не image-only.','small')
table(['Параметр','Значение'],[
 ('Обучающие запросы','5 766: 2 883 EN и их 2 883 RU-перевода'),
 ('Validation','1 216: 608 EN + 608 RU; поиск по полному каталогу'),
 ('Оптимизатор','AdamW; lr=0,001; weight decay=0,01; gradient clip=1,0'),
 ('Режим','10 эпох; batch=128; rank=64; temperature=0,07; seed=42'),
 ('Выбор','Максимум среднего EN+RU dev NDCG@5; эпоха 4: 0,012301'),
],[5.0,12.0])
p('Объектив multi-positive InfoNCE','sub')
p('Для каждого запроса loss = log Σ<sub>j∈V</sub> exp(s<sub>j</sub>/τ) − log Σ<sub>p∈P</sub> exp(s<sub>p</sub>/τ), где P - известные позитивы, V - допустимые кандидаты батча, τ=0,07. Негативы берутся из позитивных изображений других запросов батча. Все известные train-позитивы одинаковой нормализованной фразы объединяются; непомеченные соседи той же группы маскируются.')
p('Таким образом, одинаковые фразы не заставляют модель отталкивать друг от друга свои известные ответы. Полностью убрать ложные негативы нельзя: семантически подходящие картинки могут не иметь метки. Большинство 30 тысяч distractor-изображений не участвуют как негативы в каждом батче; это ограничение между обучением и полноразмерным retrieval.')
p('Эпоха 4 сохранена заранее по dev. После dev-сравнения она выбрана для русскоязычного демонстрационного режима. Перед test записаны хеши checkpoint, индекса, запросов и кода оценки. Test не используется для выбора другой эпохи или новых весов.','small')

head(7,'Метрики и протокол оценки')
p('Оценка проводится на одном каталоге из 30 914 кандидатов. Test содержит <b>680 EN-запросов и 680 соответствующих RU-переводов</b>, относящихся к 86 группам изображений. Эти языковые выборки связаны и не считаются 1 360 независимыми пользовательскими сессиями. Все known-positive цели происходят из исходных sticker-query пар; отдельной количественной оценки открыток нет.')
table(['Метрика','Определение','Зачем нужна'],[
 ('Hit@k','1, если хотя бы один известный позитив в top-k','Попал ли целевой стикер в видимую выдачу'),
 ('Recall@k','Число найденных известных позитивов / число известных позитивов','В текущей single-positive оценке совпадает с Hit@k'),
 ('NDCG@5','DCG / идеальный DCG; вес ранга r: 1/log2(r+1)','Главная метрика выбора: ранняя позиция важнее поздней'),
 ('MRR@10','1/r первого известного позитива; 0, если его нет в top-10','Показывает, насколько быстро найден целевой объект'),
 ('p50 / p95 времени','Медиана и 95-й процентиль задержки','Отдельно scoring и кодирование запроса'),
],[3.2,7.2,6.6])
p('При одном бинарном позитиве NDCG@5 равна 1/log2(r+1) для r≤5 и нулю иначе. Полный код допускает graded relevance, но оценки 0/1/2 людьми в данном эксперименте не собраны.')
p('Ограничение known-positive retrieval','sub')
p('Неизвестные альтернативы считаются нерелевантными только при вычислении benchmark. Поэтому низкое число не означает, что каждый другой кот или стикер «спасибо» плох для пользователя. И обратное: удачный скриншот не доказывает качество всего корпуса. Независимая pooled-разметка русских выдач - обязательный следующий шаг.')
p('Защита от утечек и неопределённость','sub')
p('Изображение и его переводы сохраняют один split; near-duplicate группы не разделяются. Caption строится без оценочного текста. Test-изображения присутствуют в поисковой базе, но их query-image пары не используются в обучении: это стандартный retrieval по фиксированному каталогу. Одинаковая короткая фраза может встречаться в разных split, поэтому результат не измеряет обобщение на полностью новые формулировки.')
p('Для метрик в исходных JSON: 1 000 bootstrap-повторов по группам, seed=42. Для парных разностей learned−image и RRF−image в отчёте: 10 000 повторов с совместной выборкой групп. Малое число независимых групп даёт широкие интервалы.','small')

head(8,'Dev: сравнение и диагностика')
rows=[]
for m,label in [('image','Image'),('hybrid','Hybrid 0,65/0,25/0,10'),('lexical','OCR TF-IDF'),('learned','Learned, эпоха 4')]:rows.append((label,num(metric(DEV['en'],m)),num(metric(DEV['ru'],m))))
for m,label in [('rrf','RRF 0,8/0,2'),('rrf_calibrated','RRF + calibration caption'),('rrf_calibrated_ocr','RRF + calibration + OCR'),('rrf_calibrated_all','RRF + calibration всех ветвей')]:rows.append((label,num(metric(CAL['en'],m)),num(metric(CAL['ru'],m))))
rows.append(('Native CLIP text, только EN',num(metric(arc_json(V3,'native_en_dev/metrics.json'),'image')),'не применялся'))
table(['Метод','EN NDCG@5','RU NDCG@5'],rows,[9.2,3.9,3.9])
p('Все строки относятся к dev: по 608 запросов на язык. Только три заранее зафиксированных метода затем перенесены на test. Calibration оценивала смещение ветвей по среднему train-вектору запросов; тест в подборе не участвовал.','small')
p('Почему простой hybrid ухудшился','sub')
p('На RU dev средний score по каталогу: image 0,2265, caption 0,8137, OCR 0,9409. L2-нормализация не делает распределения ветвей одинаковыми. Высокий общий фон текстовых score и ошибки описаний могут подавлять полезное визуальное ранжирование.')
p('Обнаружен caption-hub: один объект с описанием «Кванг Су. Что?» стал top-1 для 389 из 608 RU-запросов (64,0%) в caption-only. Это наблюдаемая концентрация выдачи, а не доказательство одной конкретной причины. Изменение перенормировки весов проблему не устранило; RRF помог на dev, но calibration не улучшила RRF.')
p('Нативный английский CLIP дал NDCG@5=0,0209 против 0,0197 мультиязычного baseline. Парный dev-интервал разности включает ноль. Это не поддержало гипотезу, что достаточно заменить только EN text encoder. Обученный адаптер оказался лучшим по точечной RU dev-оценке, но абсолютный результат низкий и статистически устойчивое превосходство не установлено.')

head(9,'Test: улучшение не подтверждено')
rows=[]
for l in ['en','ru']:
    for m in ['image','rrf','learned']:
        rows.append((l.upper(),m,num(metric(TEST[l],m)),num(metric(TEST[l],m,'Hit@5')),num(metric(TEST[l],m,'Hit@10')),num(metric(TEST[l],m,'MRR@10'))))
table(['Язык','Метод','NDCG@5','Hit@5','Hit@10','MRR@10'],rows,[1.6,3.2,3.1,3.0,3.0,3.1])
fig(ASSETS/'test_ci.png')
p('Рис. 3. Средние и групповые 95% bootstrap-интервалы на test; интервалы не являются гарантией качества для произвольных новых запросов.','small')
rows=[]
for lang in ['en','ru']:
    v=analysis['comparisons'][f'{lang}_learned_minus_image'];rows.append((lang.upper(),f'{v["mean"]:+.4f}',f'[{v["ci95"][0]:+.4f}; {v["ci95"][1]:+.4f}]'))
table(['Learned − image','Δ NDCG@5','Парный 95% CI'],rows,[5.3,4.0,7.7])
p('RU: baseline нашёл известный позитив в top-5 для 13 из 680 запросов, адаптер - для 9. В top-10: 18 и 19 соответственно. Значит, близкий Hit@10 не означает одинаковое качество первых позиций. EN: NDCG@5 вырос с 0,0167 до 0,0211, но интервал разности также включает ноль.')
p('<b>Вывод:</b> преимущество на RU dev не перенеслось на RU test. Возможны шум переводов, малая размеченная выборка и переобучение адаптера; данный опыт не разделяет их вклад причинно. Выбранный checkpoint сохранён. Переключать финальный метод по увиденному test и выдавать это за независимую оценку нельзя.','small')

head(10,'Поиск: примеры реальной выдачи')
p('Ниже top-1 трёх методов для одинаковых запросов и фильтров. Примеры выбраны после просмотра, чтобы показать разные типы поведения; это иллюстрации, а не случайная количественная выборка. Полная галерея содержит 18 заранее заданных сценариев × 3 метода × top-3.','small')
selected=[2,4,6,3,12,14]
rows=[]
for qi in selected:
    q=QUERIES[qi];row=[q]
    for m in ['image','rrf','learned']:
        rec=next(x for x in reviews if x['query']==q and x['method']==m);path=CACHE[rec['top3'][0]['item_id']]
        with PILImage.open(path) as im:w,h=im.size
        scale=min(95/w,64/h);row.append(Image(str(path),w*scale,h*scale))
    rows.append(row)
table(['Запрос','Image','RRF','Learned'],rows,[5.0,4.0,4.0,4.0])
p('«Хочу спать» и «обнимаю» дают содержательно подходящие картинки. Для «ничего не понимаю» адаптер выводит персонажа со знаком вопроса, а baseline - текстовый мем. Для «смеюсь до слёз» адаптер улавливает слёзы, но теряет смех: выдаёт плачущего персонажа.')
p('На «с днём рождения» baseline путает повод с поздравлением о победе; адаптер находит birthday-карточку. На «доброе утро» top-1 адаптера соответствует запросу, но в его top-3 присутствует Good Night. Качество необходимо проверять по всей видимой выдаче, включая надпись.','small')

head(11,'Категории ошибок и пределы анализа')
table(['Категория','Наблюдение','Что доработать'],[
 ('Эмоция и намерение','Смех со слезами заменён плачем; «поддержать перед экзаменом» даёт общие или чужие по языку картинки','Разметить эмоционально близкие hard negatives и короткие пользовательские ситуации'),
 ('Повод / отрицание','День рождения ↔ победа; доброе утро ↔ спокойной ночи; «God forgives, but I don’t» в ответ на «прости»','Отдельные признаки повода и языка надписи; проверка отрицания в reranker'),
 ('Составной запрос','Для робота со скрипкой в космосе находятся робот/космос либо инструмент по отдельности','Композиционные тесты; reranking по нескольким условиям; генерация по кнопке'),
 ('Шум caption/OCR','Повторяющийся caption-hub; возможные выдуманные эмоции и неточный OCR','Ручная оценка caption и CER/WER OCR; отключение низкоуверенных ветвей'),
 ('Ошибки перевода','happy hippo → счастливый хиппи; смена намерений при валидном JSON','Проверить русские переводы во всех split; версия датасета и отдельный RU gold'),
 ('Неполная разметка','Другой подходящий кот может получать 0 при единственном известном позитиве','Pooled-оценка нескольких систем людьми с градациями 0/1/2'),
],[3.2,7.0,6.8])
p('Визуальные наблюдения сделаны ассистентом по самим картинкам, а не только по VLM-caption. Они раскрывают характер ошибок, но не дают точной частоты каждой категории. В проекте нет независимого human-оценщика, согласованности между оценщиками или полноценной оценки no-match.')
p('План независимой проверки','sub')
p('Собрать 100-200 новых русских запросов: эмоции, действия, надписи, поводы открыток, отрицания и запросы без ответа. Объединить top-k разных методов, скрыть название метода и поставить оценки 0/1/2. Для части запросов привлечь второго оценщика. Порог отказа и ранжирование настраивать по новой dev-части; новую test-часть открыть один раз после фиксации решения.')
p('Нельзя утверждать, что низкие known-positive метрики полностью объясняются неполной разметкой: реальные семантические ошибки также видны. Аналогично, несколько удачных примеров не отменяют отрицательного результата количественного эксперимента.','small')

head(12,'Генерация и удаление фона')
p('Русский prompt переводится Qwen2.5-7B-Instruct [7] на английский; добавляется инструкция одиночного cartoon sticker без надписей. Готовая FLUX.1-schnell [8]: BF16, 768×768, 4 шага, guidance=0; затем U2Net через rembg [10] на CPU. Генератор и сегментатор не дообучались.')
fig(ASSETS/'generation_pairs.png')
p('Рис. 4. Пять реальных запусков. Верхний ряд - исходники; нижний - RGBA на тёмном фоне для проверки маски. Порядок соответствует таблице.','small')
table(['№ / сюжет','Время, с','Визуальное наблюдение'],[
 ('1. Кот с кофе',f'{GEN[0]["seconds"]:.2f}','Кот, улыбка и чашка сохранены; тонкие усы требуют проверки в полном размере'),
 ('2. Панда с сердцем',f'{GEN[1]["seconds"]:.2f}','Объект и сердце сохранены; видна светлая стикерная обводка'),
 ('3. Сонная сова',f'{GEN[2]["seconds"]:.2f}','Сонная поза и колпак читаются; фон удалён'),
 ('4. Робот со скрипкой',f'{GEN[3]["seconds"]:.2f}','Есть робот и инструмент, но космос не выражен; маска делает часть корпуса полупрозрачной'),
 ('5. Облачко с радугой',f'{GEN[4]["seconds"]:.2f}','Основные объекты сохранены; радуга толще, чем просили'),
],[4.4,2.2,10.4])
p('Все пять PNG имеют alpha от 0 до 255 и проверенные SHA-256. Этот технический тест исключает постоянную маску, но не измеряет качество сегментации. Особенно заметен дефект белых деталей робота. Нужна проверка foreground и границ; возможны matting или другой сегментатор, но их эффект в данном опыте не измерен.')
p('Время 14,02-14,80 с взято из generate(): включает перевод, загрузку моделей, генерацию и удаление фона; не включает запуск процесса/импорт библиотек и браузер. Это пять последовательных запусков на одном устройстве, не SLA и не замер под нагрузкой.','small')

head(13,'Скорость, ресурсы и воспроизводимость')
rows=[]
for m in ['image','rrf','learned']:
    t=TEST['ru']['methods'][m]['search_latency_ms'];g=analysis['warm_latency'][m]['warm_end_to_end_ms']
    rows.append((m,f'{t["p50"]:.1f} / {t["p95"]:.1f}',f'{g["p50"]:.1f} / {g["p95"]:.1f}'))
table(['Метод','Test scoring, мс p50 / p95','Тёплый запрос, мс p50 / p95'],rows,[4,6.5,6.5])
p('Scoring: 680 RU test-запросов с заранее рассчитанными векторами. Тёплый запрос: 18 RU-сценариев, CPU, 8 потоков, text encoder + scoring. Загрузка изображений, HTML, сеть и cold start исключены. Проектная цель <1 с выполняется для измеренной вычислительной части; полный интерфейс и параллельная нагрузка отдельно не измерялись.','small')
p('Точный поиск оправдан размером базы: одна FP32-матрица 30 914×512 занимает 63,31 MB (60,38 MiB), три - около 190 MB без накладных расходов. Это расчёт размера массивов, не измеренный peak RSS/VRAM. ANN пока не нужен для подтверждённой задержки; для роста каталога следует сравнить ANN recall и latency.')
table(['Компонент','Фактическая среда'],[
 ('Python / CUDA / GPU','3.10.12 / CUDA 12.8 / 4 × NVIDIA A100 80GB PCIe'),
 ('PyTorch','2.9.1+cu128; CUDA подтверждена тестовой операцией'),
 ('Transformers / ST','4.57.1 / sentence-transformers 5.1.2'),
 ('NumPy / sklearn / Pillow','2.2.6 / 1.7.2 / 11.3.0'),
 ('Diffusers / rembg / ONNX','0.35.1 / 2.0.67 / onnxruntime 1.23.2'),
 ('Интерфейс','Streamlit 1.50.0'),
],[5.2,11.8])
p('На поде нет внешнего интернета. Данные и закреплённые модели загружены на ноутбуке, переданы через YT и восстановлены локально. Индекс и обучение читают локальные файлы. HF_HUB_OFFLINE и TRANSFORMERS_OFFLINE включены. Корпус, архивы и модели требуют существенно больше места, чем один индекс; 147 GB раздел заполнялся при одновременном хранении архивов и распакованных весов.')
p('Четыре GPU используются для распределённой аннотации; обучение малых адаптеров и генерация не требуют четырёх GPU одновременно. Seed фиксируется, но точная побитовая повторяемость между разными GPU/библиотеками не заявляется.','small')

head(14,'Выводы и следующая итерация')
p('<b>Что удалось.</b> Собран и описан корпус, выполнены контроль файлов и групповой split, получены caption/OCR, подготовлены EN/RU-запросы, обучена retrieval-модель, сравнены альтернативы, зафиксирован выбор до test. Реальная генерация и удаление фона выполнены пять раз. Сохранены исходники, RGBA и параметры.')
p('<b>Что не подтверждено.</b> Дообучение не дало устойчивого улучшения русского поиска на test. Простое смешивание текстовых и визуальных score ухудшает dev; calibration и замена EN text encoder не решили проблему. Независимая русская relevance-оценка, OCR accuracy, автоматический no-match, качество открыток и нагрузочный тест отсутствуют.')
table(['Приоритет','Следующее действие','Как проверить'],[
 ('1. Разметка','Новые RU-запросы + слепой pooled review; аудит переводов','Отдельный новый test, NDCG с несколькими релевантными ответами; согласие оценщиков'),
 ('2. Обучение','Hard negatives из полного каталога; регуляризация и меньший lr','Сравнение по новой dev; несколько seed; без повторной подгонки текущего test'),
 ('3. Признаки','Отдельный OCR; reranker для эмоции, отрицания и повода','Ablation по типам запросов и CER/WER на транскрипциях'),
 ('4. Покрытие','Больше разных открыток и допустимых стилей; анимация отдельным потоком','Баланс сценариев, группировка вариантов, отдельные метрики по типам'),
 ('5. Генерация','Проверка маски/тонких деталей; явный контроль отсутствующих условий','Тестовые пары original/alpha с ручными масками или экспертной оценкой'),
],[2.4,7.0,7.6])
p('Для защиты','sub')
p('Показать работающий поиск, сравнить baseline и адаптер на одинаковом запросе, объяснить расхождение dev/test и неполноту меток. Затем показать генерацию и прозрачность, включая дефект робота. На вопросы отвечать по фактическим артефактам: архитектура, loss, split, метрики, причины ограничений и план проверки гипотез.')
p('Итог - исследовательский прототип с воспроизводимым отрицательным результатом по основной гипотезе улучшения RU retrieval. Его сильная сторона - полнота проверяемого процесса; основная задача следующей версии - качество разметки и оценки, затем обучение. Текст и код подготовлены с помощью AI; успешная синхронная защита этим отчётом не подтверждается.','small')

head(15,'Источники и карта артефактов')
refs=[
 ('1','nyuuzyou/stickers: источник статичных PNG','https://huggingface.co/datasets/nyuuzyou/stickers'),
 ('2','StickerQueries: человеческие query-image пары','https://huggingface.co/datasets/metchee/sticker-queries'),
 ('3','Synthetic Greeting Cards','https://huggingface.co/datasets/gauravs101/synthetic-greeting-cards'),
 ('4','OpenMoji: графика и CC BY-SA 4.0','https://github.com/hfg-gmuend/openmoji'),
 ('5','Multilingual CLIP text encoder','https://huggingface.co/sentence-transformers/clip-ViT-B-32-multilingual-v1'),
 ('6','Qwen2.5-VL-7B-Instruct','https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct'),
 ('7','Qwen2.5-7B-Instruct, перевод generation prompt','https://huggingface.co/Qwen/Qwen2.5-7B-Instruct'),
 ('8','FLUX.1-schnell','https://huggingface.co/black-forest-labs/FLUX.1-schnell'),
 ('9','CLIP, исходная архитектура','https://github.com/openai/CLIP'),
 ('10','rembg: удаление фона, U2Net','https://github.com/danielgatis/rembg'),
]
for no,label,url in refs:p(f'<b>[{no}] {escape(label)}</b><br/><link href="{url}" color="#3554B5">{escape(url)}</link>','small')
p('Карточки источников проверены 28.09.2026. Ревизии моделей закреплены полными commit hash в configs/model_revisions.json; состав данных - evidence/full_catalog/manifest.jsonl. Источники чисел: сохранённые логи и JSON проекта, а не показатели из model cards.','small')
table(['Артефакт в комплекте','Что подтверждает'],[
 ('evidence/final_v1/final_results.zip','Замороженный выбор, test EN/RU, 18 поисковых сценариев, 5 генераций, среда и логи'),
 ('evidence/full_v2/retrieval_diagnostics.zip','Полные dev-метрики, история обучения и диагностика ветвей'),
 ('evidence/full_v2/retrieval_v3_results.zip','Calibration и native CLIP dev-сравнение'),
 ('scripts/build_final_report.py','Пересчёт парных CI и построение этого PDF из сохранённых результатов'),
 ('docs/DEFENSE.md; scripts/run_demo.sh','План защиты, вопросы и команда запуска интерфейса'),
],[8.0,9.0],small=True)
p('Запуск на готовом поде: <b>bash scripts/run_demo.sh</b>. Восстановление отчёта: <b>python scripts/build_final_report.py</b>. Полная подготовка и перенос: README.md и docs/OFFLINE_TRANSFER.md. Внешний репозиторий ещё не опубликован; ссылка добавляется в ЛМС после публикации кода.','small')

def footer(canvas,doc):
    canvas.setStrokeColor(colors.HexColor('#DCE3EF'));canvas.line(2*cm,1.5*cm,19*cm,1.5*cm)
    canvas.setFont('DV',7.5);canvas.setFillColor(colors.HexColor('#65718A'))
    canvas.drawString(2*cm,1.03*cm,'Подбор стикеров и открыток · этап 2 · 28.09.2026')
    canvas.drawRightString(19*cm,1.03*cm,f'{doc.page} / 15')

target=OUT/'final_report.pdf'
doc=SimpleDocTemplate(str(target),pagesize=(21*cm,29.7*cm),leftMargin=2*cm,rightMargin=2*cm,topMargin=1.65*cm,bottomMargin=1.85*cm,title='Интеллектуальная система подбора стикеров и открыток. Этап 2',author='Р. Гулямов')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(target)
