"""SigLIP 2 search with separate image generation and background removal."""
import os
import time
from pathlib import Path

import streamlit as st

from sticker_search.siglip.search import SiglipSearch

st.set_page_config(page_title="Поиск стикеров · SigLIP 2",page_icon="💬",layout="wide")
st.title("Стикер по настроению")
run = os.environ.get("STICKER_SIGLIP_RUN","/tmp/cv_project/siglip2_v1")
catalog = os.environ.get("STICKER_CATALOG","data/catalog")
device = os.environ.get("STICKER_SIGLIP_DEVICE","cuda")
if not (Path(run)/"selection.json").exists():
    st.info("Сначала дождись оценки исходной модели и завершения подготовки индекса.")
    st.stop()


@st.cache_resource
def resources(run,catalog,device,which,selection_stamp):
    return SiglipSearch(run,catalog,device,which)


which = st.sidebar.selectbox("Модель",["selected","baseline"],format_func=lambda x:
    "Выбранная по валидации" if x=="selected" else "Исходный SigLIP 2")
index = resources(run,catalog,device,which,(Path(run)/"selection.json").stat().st_mtime_ns)
with st.form("search"):
    query = st.text_input("Что хочется выразить?","весёлый кот с чашкой кофе")
    kind = st.selectbox("Что искать",["Всё","Стикеры","Открытки"])
    use_ocr = st.checkbox("Учитывать совпадения с надписями",value=False)
    submitted = st.form_submit_button("Найти",type="primary")
if submitted and query.strip():
    start = time.perf_counter()
    results = index.search(query,kind={"Всё":None,"Стикеры":"sticker","Открытки":"postcard"}[kind],
                           method="image_ocr_rrf" if use_ocr else "image")
    st.caption(f"{len(index.items):,} изображений · {time.perf_counter()-start:.2f} с")
    columns = st.columns(4)
    for i,(row,score) in enumerate(results):
        path = Path(catalog)/row["image_path"]
        with columns[i%4]:
            st.image(str(path),width="stretch")
            with st.expander("Описание и источник"):
                st.write(row.get("caption_text",""))
                if row.get("ocr_text"):
                    st.write("Надпись: "+row["ocr_text"])
                st.caption(row.get("source",""))
            st.download_button("Скачать",path.read_bytes(),file_name=path.name,key=row["item_id"])

st.divider()
from sticker_search.generation_ui import render_generation_panel

render_generation_panel()
