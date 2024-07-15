import argparse
import torch
from collections import Counter

from bilstmTrain import (
    read_data, read_test_data, pad_sequences, load_pretrained_embeddings, build_char_vocab,
    encode_char_data, pad_char_sequences, build_prefix_suffix_vocab, encode_prefix_suffix_data,
    BiLSTM, CharLSTM, BiLSTMWithChar, BiLSTMWithPretrained, BiLSTMWithCharAndWord
)


# Function to predict and save
def predict_and_save(model, sentences, vocab, tag_set, output_file, encode_char_func=None, char_vocab=None,
                     encode_prefix_suffix_func=None, prefix_vocab=None, suffix_vocab=None):
    model.eval()
    if encode_char_func and char_vocab:
        encoded_char_sentences = encode_char_func(sentences, char_vocab)
        padded_char_sentences = pad_char_sequences(encoded_char_sentences, [char_vocab["<PAD>"]] * 15)
        X_char_test = torch.tensor(padded_char_sentences, dtype=torch.long)

    if encode_prefix_suffix_func and prefix_vocab and suffix_vocab:
        encoded_prefix_sentences, encoded_suffix_sentences = encode_prefix_suffix_func(sentences, prefix_vocab,
                                                                                       suffix_vocab)
        padded_prefix_sentences = pad_sequences(encoded_prefix_sentences, prefix_vocab["<PAD>"])
        padded_suffix_sentences = pad_sequences(encoded_suffix_sentences, suffix_vocab["<PAD>"])
        X_prefix_test = torch.tensor(padded_prefix_sentences, dtype=torch.long)
        X_suffix_test = torch.tensor(padded_suffix_sentences, dtype=torch.long)

    encoded_sentences = [[vocab.get(word, vocab["<UNK>"]) for word in sentence] for sentence in sentences]
    padded_sentences = pad_sequences(encoded_sentences, vocab["<PAD>"])
    X_test = torch.tensor(padded_sentences, dtype=torch.long)

    with torch.no_grad():
        if encode_char_func and char_vocab:
            if encode_prefix_suffix_func and prefix_vocab and suffix_vocab:
                outputs = model(X_char_test, X_test, X_prefix_test, X_suffix_test)
            elif isinstance(model, BiLSTMWithCharAndWord):
                outputs = model(X_char_test, X_test)
            else:
                outputs = model(X_char_test)
        else:
            if encode_prefix_suffix_func and prefix_vocab and suffix_vocab:
                outputs = model(X_test, X_prefix_test, X_suffix_test)
            else:
                outputs = model(X_test)

        predictions = outputs.argmax(dim=2).tolist()

    idx_to_tag = {idx: tag for tag, idx in tag_set.items()}

    with open(output_file, 'w') as f:
        for sentence, pred in zip(sentences, predictions):
            for word, tag_idx in zip(sentence, pred):
                f.write(f"{word} {idx_to_tag[tag_idx]}\n")
            f.write("\n")


def main_predict(model_choice, model_file, input_file, output_file='predictions.ner', train_file='ner/train',):
    # Load data
    train_sentences, train_tags = read_data(train_file)
    test_sentences = read_test_data(input_file)

    # Load vocabulary and tag set
    word_counts = Counter([word for sentence in train_sentences for word in sentence])
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    tag_set = {tag: idx for idx, tag in enumerate(set([tag for tags in train_tags for tag in tags]))}

    if model_choice == 'a':
        # Define BiLSTM model
        vocab_size = len(vocab)
        tagset_size = len(tag_set)
        embedding_dim = 128
        hidden_dim = 256
        model = BiLSTM(vocab_size, tagset_size, embedding_dim, hidden_dim)
    elif model_choice == 'b':
        # Define BiLSTM with Character-level LSTM model
        char_vocab = build_char_vocab(train_sentences)
        char_vocab_size = len(char_vocab)
        vocab_size = len(vocab)
        tagset_size = len(tag_set)
        char_embedding_dim = 30
        char_hidden_dim = 50
        embedding_dim = 128
        hidden_dim = 256
        model = BiLSTMWithChar(char_vocab_size, vocab_size, tagset_size, char_embedding_dim, char_hidden_dim,
                               embedding_dim, hidden_dim)
    elif model_choice == 'c':
        # Define BiLSTM with Pretrained Embeddings and Prefix-Suffix Embeddings model
        word_to_idx, embedding_matrix = load_pretrained_embeddings('vocab.txt', 'wordVectors.txt')
        prefix_vocab, suffix_vocab = build_prefix_suffix_vocab(train_sentences)
        prefix_vocab_size = len(prefix_vocab)
        suffix_vocab_size = len(suffix_vocab)
        tagset_size = len(tag_set)
        prefix_suffix_embedding_dim = 50
        hidden_dim = 256
        model = BiLSTMWithPretrained(len(word_to_idx), embedding_matrix, prefix_vocab_size, suffix_vocab_size,
                                     tagset_size, prefix_suffix_embedding_dim, hidden_dim)
    elif model_choice == 'd':
        # Define BiLSTM with Character-level LSTM and Word Embedding model
        char_vocab = build_char_vocab(train_sentences)
        char_vocab_size = len(char_vocab)
        vocab_size = len(vocab)
        tagset_size = len(tag_set)
        char_embedding_dim = 30
        char_hidden_dim = 50
        word_embedding_dim = 100
        hidden_dim = 256
        model = BiLSTMWithCharAndWord(char_vocab_size, vocab_size, tagset_size, char_embedding_dim, char_hidden_dim,
                                      word_embedding_dim, hidden_dim)

    # Load the trained model and additional data
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint['vocab']
    tag_set = checkpoint['tag_set']
    if model_choice == 'b' or model_choice == 'd':
        char_vocab = checkpoint['char_vocab']
    if model_choice == 'c':
        prefix_vocab = checkpoint['prefix_vocab']
        suffix_vocab = checkpoint['suffix_vocab']

    print(f"Model loaded from {model_file}")

    # Predict and save the results
    if model_choice == 'a':
        predict_and_save(model, test_sentences, vocab, tag_set, output_file)
    elif model_choice == 'b':
        predict_and_save(model, test_sentences, vocab, tag_set, output_file, encode_char_func=encode_char_data,
                         char_vocab=char_vocab)
    elif model_choice == 'c':
        predict_and_save(model, test_sentences, vocab, tag_set, output_file,
                         encode_prefix_suffix_func=encode_prefix_suffix_data, prefix_vocab=prefix_vocab,
                         suffix_vocab=suffix_vocab)
    elif model_choice == 'd':
        predict_and_save(model, test_sentences, vocab, tag_set, output_file, encode_char_func=encode_char_data,
                         char_vocab=char_vocab)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('repr', choices=['a', 'b', 'c', 'd'], help="Model representation: a, b, c, or d")
    parser.add_argument('modelFile', help="File to load the trained model from")
    parser.add_argument('inputFile', help="Blind input file to tag")
    parser.add_argument('--outputFile', default='predictions.pos', help="File to save the predictions")
    parser.add_argument('--trainFile', default='pos/train', help="Train file")
    args = parser.parse_args()

    main_predict(args.repr, args.modelFile, args.inputFile, args.outputFile, train_file=args.trainFile)
