---
license: apache-2.0
task_categories:
  - image-classification
  - image-to-text
  - text-to-image
language:
  - en
tags:
  - greeting-cards
  - synthetic
  - multimodal
  - illustration
  - cards
pretty_name: Synthetic Greeting Cards Dataset (Synth-GCD)
size_categories:
  - 1K<n<10K
---

# Synthetic Greeting Cards Dataset (Synth-GCD)

A modern, fully open synthetic alternative to the proprietary **Greeting Cards Dataset (GCD)** described in the paper  
*"Weakly Supervised Annotations for Multi-modal Greeting Cards Dataset"*.

This dataset contains high-quality AI-generated greeting card illustrations with corresponding short messages, designed for research in multimodal classification, retrieval, and generation.

## Dataset Summary

| Property              | Value                          |
|-----------------------|--------------------------------|
| Total Images          | ~100                           |
| Image Style           | Modern flat / watercolor / cartoon illustration |
| Text                  | Short cover messages embedded in the images |
| Categories            | Holidays, Messages, Special Occasions (and related themes) |
| License               | Apache 2.0                     |
| Generation Models     | Flux / SDXL-style models       |
| Fully Open            | Yes – you own the data         |

## Motivation

The original GCD dataset (3,700+ images) is proprietary and not publicly available.  
This synthetic version was created so researchers can freely experiment with multimodal greeting-card understanding without legal or access restrictions.

## Dataset Structure

```
synth_gcd/
├── synth_gcd_50/          # First batch (numbered + randomized names)
│   ├── 01_christmas.jpg
│   ├── 02_valentines.jpg
│   ├── ...
│   ├── lm1.jpg ... lm10.jpg
│   └── ...
├── synth_gcd_extra/       # Second batch (grouped naming)
│   ├── sm1.jpg ... sm10.jpg
│   ├── em1.jpg ... em10.jpg
│   ├── pm1.jpg ... pm10.jpg
│   ├── km1.jpg ... km10.jpg
│   └── extra1.jpg, extra2.jpg
└── README.md
```

## Theme Coverage

The images cover a wide range of greeting card themes, including (but not limited to):

**Holidays & Seasons**
- Christmas, New Year, Easter, Halloween, Diwali, Thanksgiving, Winter Greetings

**Special Occasions**
- Birthday, Anniversary, Wedding, Graduation, New Baby, Retirement, Housewarming, Engagement

**Messages & Sentiments**
- Thank You, Thinking of You, Friendship, Encouragement, Get Well, Sorry, Good Luck, Farewell
- New Beginnings, Hope, Peace, Strength, Kindness, Growth, Journey

## How the Data Was Generated

- All images were generated using modern open text-to-image models.
- Prompts were carefully crafted to produce **greeting-card style** illustrations (flat, watercolor, soft cartoon).
- Each image includes a short, natural cover message appropriate for the theme.
- No real commercial greeting cards were scraped or copied.

## Intended Uses

- Multimodal classification (image + text)
- Image-to-text / text-to-image research
- Style transfer and illustration generation
- Dataset construction and benchmarking for greeting-card domains
- Educational and research purposes

## Limitations

- Synthetic data – visual style is consistent with modern AI illustration rather than traditional printed cards.
- Text is rendered inside the image (not provided as separate text files).
- Relatively small scale compared to the original proprietary GCD (~100 images vs 3,700+).

## License

**Apache License 2.0**

You are free to use, modify, and distribute this dataset for research and commercial purposes, provided you include the original license and attribution.

## Citation

If you use this dataset, please consider citing the original GCD paper for context:

```bibtex
@article{gcd2022,
  title={Weakly Supervised Annotations for Multi-modal Greeting Cards Dataset},
  year={2022}
}
```

And optionally mention this synthetic version as:

> Synthetic Greeting Cards Dataset (Synth-GCD) – open alternative generated for research reproducibility.

## Contact / Contributions

Feel free to open issues or pull requests if you expand the dataset with more categories, higher resolution, or separate text annotations.
