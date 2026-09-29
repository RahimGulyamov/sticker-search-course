"""Run with: streamlit run app.py. Paths are configured through STICKER_* env vars."""
import os
import time
from pathlib import Path

import streamlit as st

from sticker_search.features import text_encoder
from sticker_search.retrieval import SearchIndex

st.set_page_config(page_title="Стикер по настроению",page_icon="💬",layout="wide")
st.title("Стикер по настроению")
st.write("Опиши, что хочешь сказать, — найдём стикер или открытку.")
catalog=Path(os.environ.get("STICKER_CATALOG","data/catalog"))
index_path=os.environ.get("STICKER_INDEX","runs/index")
checkpoint=os.environ.get("STICKER_CHECKPOINT")


@st.cache_resource
def resources(index_path,checkpoint):
    return SearchIndex(index_path,checkpoint),text_encoder("cpu")


if not (Path(index_path)/"vectors.npz").exists():
    st.info("Каталог ещё готовится. Запусти подготовку индекса по инструкции README.")
    st.stop()
index,encoder=resources(index_path,checkpoint)
with st.form("search"):
    query=st.text_input("Что хочется выразить?",value="поддержать друга перед экзаменом")
    kind_label=st.selectbox("Что искать",["Всё","Стикеры","Открытки"])
    labels={"По изображению":"image","Изображение и описание":"rrf","Изображение + описание + надпись":"hybrid","По надписи":"lexical"}
    if checkpoint:
        labels["Выбранная модель"]="learned"
    method_label=st.selectbox("Способ поиска",list(labels),index=len(labels)-1 if checkpoint else 0)
    submitted=st.form_submit_button("Найти",type="primary")
if submitted and query.strip():
    start=time.perf_counter()
    q=encoder.encode([query],normalize_embeddings=True)
    scores=index.scores(q,labels[method_label],[query])[0]
    kind={"Всё":None,"Стикеры":"sticker","Открытки":"postcard"}[kind_label]
    results=index.topk(scores,k=12,kind=kind)
    st.caption(f"{len(index.items):,} изображений · поиск за {time.perf_counter()-start:.2f} с")
    if not results or (labels[method_label]=="lexical" and all(score<=0 for _,score in results)):
        st.info("Совпадений не найдено. Попробуй другое описание или создай новый стикер ниже.")
    else:
        columns=st.columns(4)
        for i,(row,score) in enumerate(results):
            with columns[i%4]:
                path=catalog/row["image_path"]
                st.image(str(path),width="stretch")
                st.caption(f"Сходство: {score:.3f}")
                with st.expander("Описание и источник"):
                    st.write(row.get("caption_text") or "Описание пока не добавлено")
                    if row.get("ocr_text"):
                        st.write("Надпись: "+row["ocr_text"])
                    st.caption(row["source"])
                st.download_button("Скачать",path.read_bytes(),file_name=path.name,key=row["item_id"])

st.divider()
from sticker_search.generation_ui import render_generation_panel

render_generation_panel()
