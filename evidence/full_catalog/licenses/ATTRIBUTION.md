# Attribution and scope

## StickerQueries
Authors: Heng Er Metilda Chee, Jiayin Wang, Zhiqiang Guo, Weizhi Ma, Min Zhang.
Small Stickers, Big Meanings: A Multilingual Sticker Semantic Understanding Dataset with a Gamified Approach. ACM Multimedia 2025.
https://doi.org/10.1145/3746027.3758310
https://huggingface.co/datasets/metchee/sticker-queries
Revision: c28a358226990ef183703bb3797c2b4c5b59e73c
The dataset card declares MIT. The snapshot contains no separate license file.
Underlying third-party artwork/character rights have not been independently audited.
Original README is retained in data/raw/README.md. The complete sticker image
corpus is not redistributed in the deliverable archive; scripts retrieve originals.
Report contact sheets are research illustrations with attribution.

## OpenMoji
OpenMoji is a project of HfG Schwaebisch Gmuend, Benedikt Gross, Daniel Utz,
students and external contributors. Per-icon authors are recorded in manifest.jsonl.
https://github.com/hfg-gmuend/openmoji
Revision f9fc506a3f913be9897ab0181d611d4c910a4104 (17.0.0).
Graphics: CC BY-SA 4.0. https://creativecommons.org/licenses/by-sa/4.0/
Changes: source SVG graphics arranged on new postcard backgrounds with optional
Russian text. Derived postcard graphics: CC BY-SA 4.0.

## Tesseract and tessdata_fast
https://github.com/tesseract-ocr/tesseract
https://github.com/tesseract-ocr/tessdata_fast
Apache-2.0. Traineddata revision is recorded in data/tessdata_revision.txt.
The engine is supplied by the runtime. Weights are downloaded by analyze.py.

## Other tools
Python, NumPy, pandas, Pillow, Matplotlib, scikit-learn, CairoSVG, ReportLab,
DejaVu Sans. Installed versions are recorded in analysis/environment.json.
No generative-image model was used for the postcard dataset. SVG rendering
preserves exact Russian text for a controlled OCR diagnostic.
