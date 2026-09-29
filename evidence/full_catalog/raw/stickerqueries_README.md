---
license: mit
task_categories:
- text-generation
language:
- zh
- en
---

# StickerQueries 🧷🗨️

**StickerQueries** is a multilingual dataset for sticker query generation and retrieval. It features human-annotated query-sticker pairs that capture the expressive, emotional, and cultural semantics embedded in stickers.
So far, the StickerQueries family has been downloaded over 200+ times!

## Dataset Structure

- `stickers_queries_zh_released.csv`: Chinese sticker annotations.
- `stickers_queries_en_released.csv`: English sticker annotations.
- `stickers/`: Sticker images in `.gif`, `.png`, or `.webm` formats.

Each row in the CSV files includes:
- `sticker_id`: The file path to the corresponding sticker image.
- `labeled_queries`: A comma-separated list of sticker queries representing the intended emotion, tone, or expression.

## Annotation Process

- Each annotation was reviewed by at least **two people**.
- In total, **42 English** and **18 Chinese** annotators contributed, with **over 60 hours** spent ensuring high-quality and diverse expressions.

## Looking for a sticker query generator?

- 🈶 **Chinese Model**: [Sticker Query Generator ZH](https://huggingface.co/metchee/sticker-query-generator-zh)  
- 🇬🇧 **English Model**: [Sticker Query Generator EN](https://huggingface.co/metchee/sticker-query-generator-en)

## Large-scale Sticker Dataset

Explore the broader dataset: [U-Sticker](https://huggingface.co/datasets/metchee/u-sticker)

## Update: V2
We are working on an extended version and will update our labels into `_extended` folders/csv(s). Do stay tuned!

---

## Citation

Our paper has been accepted at ACM Multimedia 2025!:-) If you find our work helpful, please do cite us with the following,

```bibtex
@inproceedings{10.1145/3746027.3758310,
  author = {Chee, Heng Er Metilda and Wang, Jiayin and Guo, Zhiqiang and Ma, Weizhi and Zhang, Min},
  title = {Small Stickers, Big Meanings: A Multilingual Sticker Semantic Understanding Dataset with a Gamified Approach},
  year = {2025},
  isbn = {9798400720352},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3746027.3758310},
  doi = {10.1145/3746027.3758310},
  abstract = {Stickers, though small, are a highly condensed form of visual expression, ubiquitous across messaging platforms and embraced by diverse cultures, genders, and age groups. Despite their popularity, sticker retrieval remains an underexplored task due to the significant human effort and subjectivity involved in constructing high-quality query-sticker datasets. Although large language models (LLMs) excel at general NLP tasks, they falter when confronted with the nuanced, intangible, and highly specific nature of sticker query generation. To address the challenge of collecting diverse and contextually appropriate sticker search queries, we introduce Sticktionary, a gamified annotation framework designed to elicit high-quality, semantically rich queries from contributors. Using this framework, we construct StickerQueries, a multilingual dataset comprising 1,115 English and 615 Chinese sticker search queries, annotated by over 60 contributors across more than 60 hours of annotation. We demonstrate the utility of StickerQueries in several downstream tasks, including query generation and sticker retrieval. Through comprehensive quantitative and qualitative evaluations, we demonstrate that StickerQueries significantly improves the quality of query generation, retrieval accuracy, and semantic understanding within the sticker domain. To support future research, we publicly release the multilingual StickerQueries dataset and two fine-tuned query generation models. Additional experiments and supplementary case studies can be found here.},
  booktitle = {Proceedings of the 33rd ACM International Conference on Multimedia},
  pages = {13457–13463},
  numpages = {7},
  keywords = {heng er metilda chee, jiayin wang, min zhang, weizhi ma, zhiqiang guo},
  location = {Dublin, Ireland},
  series = {MM '25}
}
```