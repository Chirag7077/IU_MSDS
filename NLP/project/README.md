# Environment information
* Python version: 3.5.3
* Exported environment file: nlp_env.yml 
* Must create sub-directory "glove" and download Glove vectors pre-trained on Twitter [glove.twitter.27B.zip](https://nlp.stanford.edu/projects/glove/)

# Notebooks that create input files for my models
1. [create_combined_dataset.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/create_combined_dataset.ipynb)
2. [create_pos_tag_dataset.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/create_pos_tag_dataset.ipynb)
3. [create_constituent_tree_dataset.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/create_constituent_tree_dataset.ipynb)

# Models for sentence classification (must be run first)

| Model        | Feature Set      | Notebook |
|--------------|------------------|----------|
| bi-LSTM, MLP | Word embeddings  | [Baseline_Part1_Bi-LSTM_MLP_Glove.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Baseline_Part1_Bi-LSTM_MLP_Glove.ipynb)
| LSTM, MLP    | Word embeddings  | [Baseline_Part1_LSTM_MLP_Glove.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Baseline_Part1_LSTM_MLP_Glove.ipynb)
| bi-LSTM, MLP | POS tags         | [Extension_Part1_Bi-LSTM_MLP_PosTag.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1_Bi-LSTM_MLP_PosTag.ipynb)
| CNN 1D       | Const. tree tags | [Extension_Part1_CONV1_ConstParse.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1_CONV1_ConstParse.ipynb)
| CNN 1D POS   | POS tags         | [Extension_Part1_CONV1_PosTag.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1_CONV1_PosTag.ipynb)
| bi-LSTM, MLP | Cont. tree tags  | [Extension_Part1_LSTM_MLP_ConstParse.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1_LSTM_MLP_ConstParse.ipynb)
| bi-LSTM, MLP | POS tags         | [Extension_Part1_LSTM_MLP_PosTag.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1_LSTM_MLP_PosTag.ipynb)

# Models for joint sentence classification 

## No mixed features

  | Model        | Feature Set       | Notebook |
  |--------------|-------------------|----------|
  | bi-LSTM, MLP | Word embeddings   | [Baseline_Part2_Bi-LSTM_MLP_Glove.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Baseline_Part2_Bi-LSTM_MLP_Glove.ipynb)
  | LSTM, MLP    | Word embeddings   | [Baseline_Part2_LSTM_MLP_Glove.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Baseline_Part2_LSTM_MLP_Glove.ipynb)

## Mixed features: word embeddings + POS tags

LSTM, MLP, CNN 1D, and LSTM for joint classification step: <br>
[Extension_Part1&2_LSTM_MLP_Glove_CONV1_PosTag.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1&2_LSTM_MLP_Glove_CONV1_PosTag.ipynb)

## Mixed features: word embeddings + Constituent tree tags

LSTM, MLP, CNN 1D, and LSTM for joint classification step: <br>
[Extension_Part1&2_LSTM_MLP_Glove_CONV1_ConstParse.ipynb](https://github.com/csathler/IU_MSDS/blob/master/NLP/project/Extension_Part1&2_LSTM_MLP_Glove_CONV1_ConstParse.ipynb)
