# NLP project 1: Quora Question Pairs Challenge

Goal of the [challenge](https://www.kaggle.com/c/quora-question-pairs):
Tackle this natural language processing problem by applying advanced techniques to classify whether question pairs are duplicates or not. 

# Collaborators:
- Alejandro Astruc
- Joel Dieguez
- Alba Garcia
- Clàudia Valverde

# Repositeroy Content 
- `utils.py` : This python file contains functions created by the different colaborators.
- `utils_Alejandro.ipynb` : focuses on implementing Word2Vec embedding (from scratch as well as pre-trained),  test this approach for different classifiers.
- `utils_Joel.ipynb` : this notebook has been executed in colab as it uses GPUs, it focuses in more advanced embeddings from pre-trained models, and test its approach for different classifiers (for distance similarity and direct embedding classification).
- `utils_Alba.ipynb` : data cleaning and pre-processing, implementation of different functions for basic NLP features, and it tests this basic approach on different classifiers.
- `utils_Clàudia.ipynb` : distance features, and exploration of different vectorizers. It performas ablation study of the simple approaches also with basic features from utils_Alba.ipynb.
- `train_models.ipynb` : the training of the models reported in main.pdf are performed in this notebook. All trained models and data created for giving results are stored in `models/` under its corresponding subfolder.
- `reproduce_results.ipynb` : the results reported in main.pdf are shown in this notebook.
- main.pdf : Report of the work done, stating how we have divided the work among the colaborators as well as how we have structured the project, our overall approach and the different experiments that we have implemented and tested. It can also be found in: https://www.overleaf.com/9513681351wbrhnzwnhbbk#a256e9
- `requirements.txt`: necessary packages and versions to run `reproduce_results.ipynb`.
