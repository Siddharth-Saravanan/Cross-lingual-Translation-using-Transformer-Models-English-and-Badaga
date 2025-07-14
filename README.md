<div align="center">

# Transformer-Based Cross-lingual Machine Translation for Low-resource Languages: A Case Study of Badaga

  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?style=for-the-badge&logo=hugging-face&logoColor=black" alt="Hugging Face" />
  
</div>

## 📝 Overview

This project focuses on building and evaluating a cross-lingual machine translation system for a low-resource language, Badaga, using state-of-the-art Transformer-based models. The primary goal is to address the linguistic digital divide by creating effective translation tools for underrepresented languages. This work is part of an academic research project.

## 🎯 Problem Statement

Low-resource languages like Badaga lack the extensive parallel corpora required to train large-scale neural machine translation models, hindering their inclusion in the digital world. This project aims to leverage multilingual pre-trained models to develop a reliable translation system between English and Badaga, facilitating communication and content creation for the Badaga-speaking community.

## 🛠️ Models

This project employs three pre-trained transformer-based models for the cross-lingual machine translation task.

* **mBART**: The mBART (Multilingual Bidirectional and Auto-Regressive Transformers) model is a sequence-to-sequence transformer pre-trained on monolingual data in 25 languages. For this project, the mBART model was fine-tuned on the Badaga-English parallel corpus, leveraging its multilingual capabilities.

* **mT5**: The mT5 (Multilingual T5) model is a massively multilingual variant of the T5 transformer model, pre-trained on data from 101 languages. The model''s capacity to handle multiple languages and its efficient encoder-decoder framework were utilized by fine-tuning it on the parallel dataset.

* **MarianMT**: MarianMT is an open-source neural machine translation framework based on the Transformer architecture. The MarianMT model was trained from scratch on the Badaga-English parallel corpus, leveraging its efficient training procedures and support for low-resource languages.

## 💾 Dataset

The parallel corpus for this research was sourced from the official repository for the paper **"CoRePooL: A Contrastive Representation Pooling for Sentence Embeddings"**. You can find the original dataset and related resources here: [rbg-research/CoRePooL](https://github.com/rbg-research/CoRePooL).

## 📊 Results

The models were evaluated using standard machine translation metrics. The results highlight the effectiveness of transfer learning for low-resource language translation. Detailed output logs for each model are available in the `Results` directory.

| Model | BLEU (En-Ba) | METEOR (En-Ba) | BLEU (Ba-En) | METEOR (Ba-En) | WER (En-Ba) | WER (Ba-En) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **mBART** | 0.9561 | 0.9578 | 0.9599 | 0.9672 | 0.031 | 0.028 |
| **mT5** | 0.6151 | 0.4959 | 0.6985 | 0.7671 | 0.371 | 0.263 |
| **MarianMT**| 0.8210 | 0.8665 | 0.8627 | 0.9067 | N/A | 0.1 |

## 📂 Project Structure
```
.
├── Source Code/
│   ├── MarianMT.ipynb        # Jupyter Notebook for the MarianMT model
│   ├── mBART_50_MMT.ipynb    # Jupyter Notebook for the mBART model
│   └── t5_nlp.ipynb          # Jupyter Notebook for the mT5 model
│
├── Results/
│   ├── marian_res_baen.csv   # Results for MarianMT (Badaga to English)
│   ├── marian_res_enba.csv   # Results for MarianMT (English to Badaga)
│   ├── mbart_res_baen.csv    # Results for mBART (Badaga to English)
│   ├── mbart_res_enba.csv    # Results for mBART (English to Badaga)
│   ├── t5_res_baen.csv       # Results for mT5 (Badaga to English)
│   └── t5_res_enba.csv       # Results for mT5 (English to Badaga)
```

## 🚀 How to Run

### Prerequisites

* Python 3.8+
* Jupyter Notebook or Google Colab

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Siddharth-Saravanan/your-repo-name.git](https://github.com/Siddharth-Saravanan/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install dependencies:**
    The notebooks were run in a Google Colab environment, which includes most necessary packages. The core libraries can be installed via pip if you are running locally:
    ```bash
    pip install torch transformers sentencepiece pandas numpy scikit-learn
    ```

### Execution

The repository includes Jupyter notebooks for each model implementation inside the `Source Code` directory. You can run these notebooks in your preferred environment to see the training process and evaluation.

## 📜 License

This project is released under the MIT License. It was developed as part of an academic requirement and is shared for educational and research purposes. See the `LICENSE` file for more details.
