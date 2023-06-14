# Code and Data: Proportionality as an argument

## Code and Data for our Paper

Kilian Lüders & Bent Stohlmann: *Proportionality as an argument. Identification of a judicial decision technique.*

14.6.2023 (Draft Version)

Contact: kilian.lueders@hu-berlin.de

## Abstract
Constitutional courts are important and powerful actors in many democracies. In particular, the German Federal Constitutional Court (GFCC) is internationally known not only for its strength as a national institution, but also as the inventor of a now internationally established decision-making technique: the proportionality test. In order to better understand the behavior of the court beyond its decision outcomes, we use argument mining approaches to classify decisions of the GFCC regarding whether they invoke proportionality or not.
Thereby, our paper makes three contributions: Firstly, it critically discusses the understanding of argument in the argument mining literature and introduces proportionality as a legal argument technique. Secondly, it presents a new dataset in which proportionality was annotated at the sentence-level in 300 decisions. Thirdly, rule-based and machine learning methods for classifying decisions are tested.

## Files

```bash
├── classification_simpletransformers.ipynb
├── classification.ipynb            # code of ML classification
├── data
│   ├── 2023_3_7_vhmk_data.csv      # annotation data
│   ├── Metadaten2.6.1.csv          # metadata on GFCC decisions
│   ├── 20230607_performance_data.csv
│   ├── training_data.pkl           # output preprocessing.py
│   └── we_model.model              # Word Embedding (not included)
├── evaluation.ipynb                # visualisations and tabels
├── explain_classifier.ipynb        # explains classifier (LIME)
├── fig                             # Figures and Tabels for paper
│   ...
├── ml-env-cpu.yaml                 # conda env
├── preprocessing.py                # creates training_data.pkl
├── README.md
└── rule_based_classification.py    # code for rule based class
```
Please note that Word Embedding Model is not in the repo due to size issues. It will be made available elsewhere soon. 

## Data
Three data resources were used for this project:

### 1. Proportionality Annotations

```
Lüders, Kilian, Wendel, Luisa, Reule, Sophie, Stohlmann, Bent, Hoeft, Leo & Tischbirek, Alexander. (2023). Proportionality. An
annotated dataset of GFCC decisions. [comming soon].
```

### 2. BVerfGE-Korpus - Decision text of the GFCC

```
Möllers, Christoph, Shadrova, Anna, & Wendel, Luisa. (2021). BVerfGE-Korpus (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.4551408
```

### 3. Metadata on the GFCC

```
Wendel, Luisa. (2023). Metadaten zu Entscheidungen des Bundesverfassungsgerichts (2.6.1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7664631
```

## Setup
For the configuration, the file *ml-env-cpu.yaml* is available that can be used to set up a Conda environment. This is a CPU-only environment, so it should run locally on (almost) all machines. If you want to re-run the models, especially the NN, make sure you have a GPU available and check which Pytorch version is compatible with your hardware. In this case you need to change your environment.