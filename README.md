# BiLSTM POS and NER Tagging (nlp course project)

This repository contains the implementation of BiLSTM models for Part-of-Speech (POS) and Named Entity Recognition (NER) tagging. It includes several variants of BiLSTM models, including those with character-level embeddings and pre-trained word embeddings.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Models](#models)
- [Functions](#functions)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/tsyrulb/BiLSTM_POS_NER.git
    cd BiLSTM_POS_NER
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training

To train a model, run the `bilstmTrain.py` script with the desired model type and dataset.

```bash
python bilstmTrain.py <model_choice> <train_file> <model_file> [--devFile <dev_file>] [--epochs <epochs>] [--batch_size <batch_size>] [--eval_interval <eval_interval>]
```

- `<model_choice>`: Model representation ('a', 'b', 'c', or 'd')
- `<train_file>`: Input file to train on
- `<model_file>`: File to save the trained model
- `--devFile`: Development file for evaluation (default: 'pos/dev')
- `--epochs`: Number of epochs for training (default: 5)
- `--batch_size`: Batch size for training (default: 32)
- `--eval_interval`: Number of sentences to process before evaluation (default: 500)

Example:
```bash
python bilstmTrain.py c pos/train model_pos_c.pth --devFile pos/dev --epochs 10 --batch_size 32 --eval_interval 500
```

### Prediction

To predict tags for a test dataset, run the `bilstmPredict.py` script with the desired model type and model file.

```bash
python bilstmPredict.py <model_choice> <model_file> <input_file> [--outputFile <output_file>] [--trainFile <train_file>]
```

- `<model_choice>`: Model representation ('a', 'b', 'c', or 'd')
- `<model_file>`: File to load the trained model from
- `<input_file>`: Blind input file to tag
- `--outputFile`: File to save the predictions (default: 'predictions.pos')
- `--trainFile`: Train file for vocabulary and tag set (default: 'pos/train')

Example:
```bash
python bilstmPredict.py c model_pos_c.pth pos/test --outputFile predictions.pos --trainFile pos/train
```

## Models

### BiLSTM Models

1. `BiLSTM`: A basic BiLSTM model.
2. `BiLSTMWithChar`: A BiLSTM model with character-level embeddings.
3. `BiLSTMWithPretrained`: A BiLSTM model with pre-trained word embeddings and prefix-suffix embeddings.
4. `BiLSTMWithCharAndWord`: A BiLSTM model with both character-level embeddings and word embeddings.

## Functions

### Training Functions

- `read_data(file_path)`: Reads data from the given file path.
- `read_test_data(file_path)`: Reads test data from the given file path.
- `pad_sequences(sequences, pad_value=0)`: Pads sequences to the maximum length.
- `load_pretrained_embeddings(vocab_file, embeddings_file)`: Loads pre-trained embeddings.
- `build_char_vocab(sentences)`: Builds a character vocabulary.
- `encode_char_data(sentences, char_vocab, max_word_len=15)`: Encodes character data.
- `pad_char_sequences(sequences, pad_value=[0])`: Pads character sequences.
- `build_prefix_suffix_vocab(sentences, prefix_len=3, suffix_len=3)`: Builds prefix and suffix vocabulary.
- `encode_prefix_suffix_data(sentences, prefix_vocab, suffix_vocab, prefix_len=3, suffix_len=3)`: Encodes prefix and suffix data.
- `calculate_accuracy(outputs, tags, tagset_size)`: Calculates accuracy of the model.

### Prediction Functions

- `predict_and_save(model, sentences, vocab, tag_set, output_file, encode_char_func=None, char_vocab=None, encode_prefix_suffix_func=None, prefix_vocab=None, suffix_vocab=None)`: Predicts tags and saves the results.
