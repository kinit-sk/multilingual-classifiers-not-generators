# Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification

This repository contains the code for the paper accepted to the findings of the EACL 2026 conference titled "Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification". The focus of the paper is comparison of using data-driven distillation through the synthetic samples to train/empower smaller LLMs with the performance of the large generator model. The main finding is that the large language models are better suited as generators of synthetic samples in low- and medium-resource languages that are then used for training of smaller models, instead of directly used for classification.

To reproduce all of the experiments, it is enough to just run the `run_experiments.sh` script (Note it may take a long time and so can be split into multiple trainings). The experiments can be extended with further models, datasets and languages by modifying the `main.py` file (and for languages adding new `.csv` files to the `data` folder in the same format).

We also provide all the data (including synthetic and human-labelled ones) in the `data` folder.


# Paper Citing

```
@inproceedings{pecher-etal-2026-better,
    title = "Better as Generators Than Classifiers: Leveraging LLMs and Synthetic Data for Low-Resource Multilingual Classification",
    author = "Pecher, Branislav  and
      Cegin, Jan and
      Belanec, Robert and
      Srba, Ivan  and
      Simko, Jakub and
      Bielikova, Maria",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2026",
    publisher = "Association for Computational Linguistics",
}
```