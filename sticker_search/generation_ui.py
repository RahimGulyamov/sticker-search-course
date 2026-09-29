"""Two explicit generation steps shared by both Streamlit applications."""
import os
import uuid
from pathlib import Path

import streamlit as st


@st.fragment
def render_generation_panel():
    """Keep outputs across widget reruns and retry cutout without running FLUX."""
    st.subheader("Создать новый стикер")
    if os.environ.get("STICKER_ENABLE_GENERATION") != "1":
        st.caption("Для генерации подключи подготовленные модели FLUX и удаления фона.")
        return

    from .generate import generate_image
    from .remove_background import remove_background

    prompt = st.text_input("Описание нового стикера", "весёлый кот с чашкой кофе", key="generation_prompt")
    if st.button("Сгенерировать изображение", key="generate_image", disabled=not prompt.strip()):
        try:
            with st.spinner("Генерируем изображение…"):
                original = generate_image(prompt, Path("runs/generated")/uuid.uuid4().hex)
            st.session_state["generation_result"] = dict(original=str(original), prompt=prompt, cutout=None)
        except Exception as error:
            st.error(f"Генерация не завершилась: {error}")

    result = st.session_state.get("generation_result")
    if not result:
        st.caption("Сначала создай изображение. Затем можно отдельно удалить фон.")
        return

    original = Path(result["original"])
    if not original.is_file():
        st.warning("Исходное изображение больше недоступно. Сгенерируй новое.")
        return
    st.caption("Запрос для показанного изображения: " + result["prompt"])
    left, right = st.columns(2)
    with left:
        st.write("Исходное изображение")
        st.image(str(original), width=384)
        st.download_button("Скачать исходник", original.read_bytes(), "original.png", "image/png",
                           key="download_original", on_click="ignore")
        if st.button("Удалить фон", key="remove_background"):
            # Clear the previous preview before attempting a new mask.
            result["cutout"] = None
            try:
                with st.spinner("Удаляем фон…"):
                    cutout = remove_background(original, original.with_name("sticker.png"))
                result["cutout"] = str(cutout)
            except Exception as error:
                st.error(f"Не удалось удалить фон: {error}")
                st.info("Исходник сохранён. Можно повторить удаление фона без новой генерации.")

    with right:
        cutout = Path(result["cutout"]) if result.get("cutout") else None
        if cutout is not None and cutout.is_file():
            st.write("Стикер без фона")
            st.image(str(cutout), width=384)
            st.download_button("Скачать PNG без фона", cutout.read_bytes(), "sticker.png", "image/png",
                               key="download_cutout", on_click="ignore")
        else:
            st.caption("Здесь появится результат после нажатия «Удалить фон».")
