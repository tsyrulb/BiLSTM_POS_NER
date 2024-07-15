__all__ = [
    "read_data", "read_test_data", "pad_sequences", "load_pretrained_embeddings", "build_char_vocab",
    "encode_char_data", "pad_char_sequences", "build_prefix_suffix_vocab", "encode_prefix_suffix_data",
    "BiLSTM", "CharLSTM", "BiLSTMWithChar", "BiLSTMWithPretrained", "BiLSTMWithCharAndWord"
]

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import numpy as np
import sys
import argparse


# Function to read data
def read_data(file_path):
    sentences, tags = [], []
    sentence, tag = [], []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    tags.append(tag)
                    sentence, tag = [], []
            else:
                word, pos = line.split()
                sentence.append(word)
                tag.append(pos)
    if sentence and tag:
        sentences.append(sentence)
        tags.append(tag)
    return sentences, tags


# Function to read test data
def read_test_data(file_path):
    sentences = []
    sentence = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                sentence.append(line)
    if sentence:
        sentences.append(sentence)
    return sentences


# Function to pad sequences
def pad_sequences(sequences, pad_value=0):
    max_length = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]


# Load pre-trained embeddings
def load_pretrained_embeddings(vocab_file, embeddings_file):
    with open(vocab_file, 'r') as f:
        vocab = [line.strip() for line in f]

    embeddings = []
    with open(embeddings_file, 'r') as f:
        for line in f:
            embeddings.append([float(x) for x in line.strip().split()])

    word_to_idx = {word: idx + 2 for idx, word in enumerate(vocab)}  # +2 to account for padding and unknown tokens
    word_to_idx["<PAD>"] = 0
    word_to_idx["<UNK>"] = 1

    embedding_matrix = np.zeros((len(vocab) + 2, len(embeddings[0])))  # +2 for padding and unknown tokens
    for idx, embedding in enumerate(embeddings):
        embedding_matrix[idx + 2] = embedding

    return word_to_idx, embedding_matrix


# Function to build character vocabulary
def build_char_vocab(sentences):
    char_vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            for char in word:
                if char not in char_vocab:
                    char_vocab[char] = len(char_vocab)
    return char_vocab


# Function to encode character data
def encode_char_data(sentences, char_vocab, max_word_len=15):
    encoded_char_sentences = []
    for sentence in sentences:
        encoded_sentence = []
        for word in sentence:
            encoded_word = [char_vocab.get(char, char_vocab['<UNK>']) for char in word]
            if len(encoded_word) > max_word_len:
                encoded_word = encoded_word[:max_word_len]
            else:
                encoded_word += [char_vocab['<PAD>']] * (max_word_len - len(encoded_word))
            encoded_sentence.append(encoded_word)
        encoded_char_sentences.append(encoded_sentence)
    return encoded_char_sentences


# Function to pad character sequences
def pad_char_sequences(sequences, pad_value=[0]):
    max_length = max(len(seq) for seq in sequences)
    return [seq + [pad_value] * (max_length - len(seq)) for seq in sequences]


# Function to build prefix and suffix vocabulary
def build_prefix_suffix_vocab(sentences, prefix_len=3, suffix_len=3):
    prefix_vocab = {'<PAD>': 0, '<UNK>': 1}
    suffix_vocab = {'<PAD>': 0, '<UNK>': 1}
    for sentence in sentences:
        for word in sentence:
            if len(word) >= prefix_len:
                prefix = word[:prefix_len]
                if prefix not in prefix_vocab:
                    prefix_vocab[prefix] = len(prefix_vocab)
            if len(word) >= suffix_len:
                suffix = word[-suffix_len:]
                if suffix not in suffix_vocab:
                    suffix_vocab[suffix] = len(suffix_vocab)
    return prefix_vocab, suffix_vocab


# Function to encode prefix and suffix data
def encode_prefix_suffix_data(sentences, prefix_vocab, suffix_vocab, prefix_len=3, suffix_len=3):
    encoded_prefix_sentences = []
    encoded_suffix_sentences = []
    for sentence in sentences:
        encoded_prefix_sentence = []
        encoded_suffix_sentence = []
        for word in sentence:
            if len(word) >= prefix_len:
                prefix = word[:prefix_len]
                encoded_prefix_sentence.append(prefix_vocab.get(prefix, prefix_vocab['<UNK>']))
            else:
                encoded_prefix_sentence.append(prefix_vocab['<PAD>'])

            if len(word) >= suffix_len:
                suffix = word[-suffix_len:]
                encoded_suffix_sentence.append(suffix_vocab.get(suffix, suffix_vocab['<UNK>']))
            else:
                encoded_suffix_sentence.append(suffix_vocab['<PAD>'])

        encoded_prefix_sentences.append(encoded_prefix_sentence)
        encoded_suffix_sentences.append(encoded_suffix_sentence)
    return encoded_prefix_sentences, encoded_suffix_sentences


# Function to calculate accuracy
def calculate_accuracy(outputs, tags, tagset_size):
    _, predicted = torch.max(outputs, -1)
    mask = (tags != -1).float()
    correct = (predicted == tags).float() * mask
    accuracy = correct.sum() / mask.sum()
    return accuracy.item()


# Define Models
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim=128, hidden_dim=256):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.bilstm1 = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out1, _ = self.bilstm1(x)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        logits = self.fc(lstm_out2)
        return logits


class CharLSTM(nn.Module):
    def __init__(self, char_vocab_size, char_embedding_dim, char_hidden_dim):
        super(CharLSTM, self).__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_embedding_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_embedding_dim, char_hidden_dim, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.char_embedding(x)
        _, (h_n, _) = self.char_lstm(x)
        word_embedding = torch.cat((h_n[0], h_n[1]), dim=1)
        return word_embedding


class BiLSTMWithChar(nn.Module):
    def __init__(self, char_vocab_size, vocab_size, tagset_size, char_embedding_dim=30, char_hidden_dim=50,
                 word_embedding_dim=100, hidden_dim=256):
        super(BiLSTMWithChar, self).__init__()
        self.char_lstm = CharLSTM(char_vocab_size, char_embedding_dim, char_hidden_dim)
        self.bilstm1 = nn.LSTM(char_hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x_char):
        batch_size, seq_len, max_word_len = x_char.size()
        x_char = x_char.view(batch_size * seq_len, max_word_len)
        word_embeddings = self.char_lstm(x_char)
        word_embeddings = word_embeddings.view(batch_size, seq_len, -1)
        lstm_out1, _ = self.bilstm1(word_embeddings)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        logits = self.fc(lstm_out2)
        return logits


class BiLSTMWithPretrained(nn.Module):
    def __init__(self, vocab_size, embedding_matrix, prefix_vocab_size, suffix_vocab_size, tagset_size,
                 prefix_suffix_embedding_dim=50, hidden_dim=256):
        super(BiLSTMWithPretrained, self).__init__()
        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=False)
        self.prefix_embedding = nn.Embedding(prefix_vocab_size, prefix_suffix_embedding_dim, padding_idx=0)
        self.suffix_embedding = nn.Embedding(suffix_vocab_size, prefix_suffix_embedding_dim, padding_idx=0)
        self.linear = nn.Linear(self.word_embedding.embedding_dim + prefix_suffix_embedding_dim * 2, hidden_dim)
        self.bilstm1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x_word, x_prefix, x_suffix):
        word_embeddings = self.word_embedding(x_word)
        prefix_embeddings = self.prefix_embedding(x_prefix)
        suffix_embeddings = self.suffix_embedding(x_suffix)

        combined_embeddings = torch.cat((word_embeddings, prefix_embeddings, suffix_embeddings), dim=2)
        combined_embeddings = self.linear(combined_embeddings)

        lstm_out1, _ = self.bilstm1(combined_embeddings)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        logits = self.fc(lstm_out2)
        return logits


class BiLSTMWithCharAndWord(nn.Module):
    def __init__(self, char_vocab_size, vocab_size, tagset_size, char_embedding_dim=30, char_hidden_dim=50,
                 word_embedding_dim=100, hidden_dim=256):
        super(BiLSTMWithCharAndWord, self).__init__()
        self.char_lstm = CharLSTM(char_vocab_size, char_embedding_dim, char_hidden_dim)
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        self.linear = nn.Linear(char_hidden_dim * 2 + word_embedding_dim, hidden_dim)
        self.bilstm1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.bilstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, x_char, x_word):
        batch_size, seq_len, max_word_len = x_char.size()
        x_char = x_char.view(batch_size * seq_len, max_word_len)
        char_embeddings = self.char_lstm(x_char)
        char_embeddings = char_embeddings.view(batch_size, seq_len, -1)

        word_embeddings = self.word_embedding(x_word)

        combined_embeddings = torch.cat((char_embeddings, word_embeddings), dim=2)
        combined_embeddings = self.linear(combined_embeddings)

        lstm_out1, _ = self.bilstm1(combined_embeddings)
        lstm_out2, _ = self.bilstm2(lstm_out1)
        logits = self.fc(lstm_out2)
        return logits


# Main function for training
def main_train(model_choice, train_file, model_file, dev_file='dev', epochs=5, batch_size=32, eval_interval=500):
    # Load data
    train_sentences, train_tags = read_data(train_file)
    dev_sentences, dev_tags = read_data(dev_file)

    # Create vocabulary and tag set
    word_counts = Counter([word for sentence in train_sentences for word in sentence])
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(word_counts.items())}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    tag_set = {tag: idx for idx, tag in enumerate(set([tag for tags in train_tags for tag in tags]))}

    # Encode data
    def encode_data(sentences, tags, vocab, tag_set):
        encoded_sentences = [[vocab.get(word, vocab["<UNK>"]) for word in sentence] for sentence in sentences]
        encoded_tags = [[tag_set[tag] for tag in tag_seq] for tag_seq in tags]
        return encoded_sentences, encoded_tags

    encoded_train_sentences, encoded_train_tags = encode_data(train_sentences, train_tags, vocab, tag_set)
    encoded_dev_sentences, encoded_dev_tags = encode_data(dev_sentences, dev_tags, vocab, tag_set)

    # Pad sequences
    padded_train_sentences = pad_sequences(encoded_train_sentences, vocab["<PAD>"])
    padded_train_tags = pad_sequences(encoded_train_tags, -1)
    padded_dev_sentences = pad_sequences(encoded_dev_sentences, vocab["<PAD>"])
    padded_dev_tags = pad_sequences(encoded_dev_tags, -1)

    # Convert to PyTorch tensors
    X_train = torch.tensor(padded_train_sentences, dtype=torch.long)
    y_train = torch.tensor(padded_train_tags, dtype=torch.long)
    X_dev = torch.tensor(padded_dev_sentences, dtype=torch.long)
    y_dev = torch.tensor(padded_dev_tags, dtype=torch.long)

    if model_choice == 'a':
        # Define BiLSTM model
        vocab_size = len(vocab)
        tagset_size = len(tag_set)
        embedding_dim = 128
        hidden_dim = 256
        model = BiLSTM(vocab_size, tagset_size, embedding_dim, hidden_dim)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create Dataset and DataLoader
        class PosDataset(Dataset):
            def __init__(self, X, y):
                self.X = X
                self.y = y

            def __len__(self):
                return len(self.X)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]

        train_dataset = PosDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        dev_dataset = PosDataset(X_dev, y_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    elif model_choice == 'b':
        # Define BiLSTM with Character-level LSTM model
        vocab_size = len(vocab)
        char_vocab = build_char_vocab(train_sentences)
        char_vocab_size = len(char_vocab)
        tagset_size = len(tag_set)
        char_embedding_dim = 30
        char_hidden_dim = 50
        embedding_dim = 128
        hidden_dim = 256
        model = BiLSTMWithChar(char_vocab_size, vocab_size, tagset_size, char_embedding_dim, char_hidden_dim,
                               embedding_dim, hidden_dim)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Encode character data
        encoded_char_train_sentences = encode_char_data(train_sentences, char_vocab)
        encoded_char_dev_sentences = encode_char_data(dev_sentences, char_vocab)

        # Pad character sequences
        padded_char_train_sentences = pad_char_sequences(encoded_char_train_sentences, [char_vocab["<PAD>"]] * 15)
        padded_char_dev_sentences = pad_char_sequences(encoded_char_dev_sentences, [char_vocab["<PAD>"]] * 15)

        # Convert to PyTorch tensors
        X_char_train = torch.tensor(padded_char_train_sentences, dtype=torch.long)
        X_char_dev = torch.tensor(padded_char_dev_sentences, dtype=torch.long)

        # Create Dataset and DataLoader for character-level input
        class CharPosDataset(Dataset):
            def __init__(self, X_char, y):
                self.X_char = X_char
                self.y = y

            def __len__(self):
                return len(self.X_char)

            def __getitem__(self, idx):
                return self.X_char[idx], self.y[idx]

        train_dataset = CharPosDataset(X_char_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        dev_dataset = CharPosDataset(X_char_dev, y_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    elif model_choice == 'c':
        # Load pre-trained embeddings
        word_to_idx, embedding_matrix = load_pretrained_embeddings('vocab.txt', 'wordVectors.txt')
        prefix_vocab, suffix_vocab = build_prefix_suffix_vocab(train_sentences)
        prefix_vocab_size = len(prefix_vocab)
        suffix_vocab_size = len(suffix_vocab)
        tagset_size = len(tag_set)
        prefix_suffix_embedding_dim = 50
        embedding_dim = 128
        hidden_dim = 256
        model = BiLSTMWithPretrained(len(word_to_idx), embedding_matrix, prefix_vocab_size, suffix_vocab_size,
                                     tagset_size, prefix_suffix_embedding_dim, hidden_dim)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Encode prefix and suffix data
        encoded_prefix_train_sentences, encoded_suffix_train_sentences = encode_prefix_suffix_data(train_sentences,
                                                                                                   prefix_vocab,
                                                                                                   suffix_vocab)
        encoded_prefix_dev_sentences, encoded_suffix_dev_sentences = encode_prefix_suffix_data(dev_sentences,
                                                                                               prefix_vocab,
                                                                                               suffix_vocab)

        # Pad prefix and suffix sequences
        padded_prefix_train_sentences = pad_sequences(encoded_prefix_train_sentences, prefix_vocab["<PAD>"])
        padded_suffix_train_sentences = pad_sequences(encoded_suffix_train_sentences, suffix_vocab["<PAD>"])
        padded_prefix_dev_sentences = pad_sequences(encoded_prefix_dev_sentences, prefix_vocab["<PAD>"])
        padded_suffix_dev_sentences = pad_sequences(encoded_suffix_dev_sentences, suffix_vocab["<PAD>"])

        # Convert to PyTorch tensors
        X_prefix_train = torch.tensor(padded_prefix_train_sentences, dtype=torch.long)
        X_suffix_train = torch.tensor(padded_suffix_train_sentences, dtype=torch.long)
        X_prefix_dev = torch.tensor(padded_prefix_dev_sentences, dtype=torch.long)
        X_suffix_dev = torch.tensor(padded_suffix_dev_sentences, dtype=torch.long)

        # Create Dataset and DataLoader for prefix-suffix input
        class PrefixSuffixPosDataset(Dataset):
            def __init__(self, X_word, X_prefix, X_suffix, y):
                self.X_word = X_word
                self.X_prefix = X_prefix
                self.X_suffix = X_suffix
                self.y = y

            def __len__(self):
                return len(self.X_word)

            def __getitem__(self, idx):
                return self.X_word[idx], self.X_prefix[idx], self.X_suffix[idx], self.y[idx]

        train_dataset = PrefixSuffixPosDataset(X_train, X_prefix_train, X_suffix_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        dev_dataset = PrefixSuffixPosDataset(X_dev, X_prefix_dev, X_suffix_dev, y_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    elif model_choice == 'd':
        # Define BiLSTM with Character-level LSTM and Word Embedding model
        char_vocab = build_char_vocab(train_sentences)
        char_vocab_size = len(char_vocab)
        tagset_size = len(tag_set)
        char_embedding_dim = 30
        char_hidden_dim = 50
        word_embedding_dim = 100
        hidden_dim = 256
        model = BiLSTMWithCharAndWord(char_vocab_size, len(vocab), tagset_size, char_embedding_dim, char_hidden_dim,
                                      word_embedding_dim, hidden_dim)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Encode character data
        encoded_char_train_sentences = encode_char_data(train_sentences, char_vocab)
        encoded_char_dev_sentences = encode_char_data(dev_sentences, char_vocab)

        # Pad character sequences
        padded_char_train_sentences = pad_char_sequences(encoded_char_train_sentences, [char_vocab["<PAD>"]] * 15)
        padded_char_dev_sentences = pad_char_sequences(encoded_char_dev_sentences, [char_vocab["<PAD>"]] * 15)

        # Convert to PyTorch tensors
        X_char_train = torch.tensor(padded_char_train_sentences, dtype=torch.long)
        X_char_dev = torch.tensor(padded_char_dev_sentences, dtype=torch.long)

        # Create Dataset and DataLoader for character-level input
        class CharPosDataset(Dataset):
            def __init__(self, X_char, X_word, y):
                self.X_char = X_char
                self.X_word = X_word
                self.y = y

            def __len__(self):
                return len(self.X_char)

            def __getitem__(self, idx):
                return self.X_char[idx], self.X_word[idx], self.y[idx]

        train_dataset = CharPosDataset(X_char_train, X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        dev_dataset = CharPosDataset(X_char_dev, X_dev, y_dev)
        dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False)

    # Training loop with evaluation every 500 sentences
    for epoch in range(epochs):
        model.train()
        processed_sentences = 0
        for batch in train_loader:
            if model_choice in ['a', 'b']:
                sentences, tags = batch
                optimizer.zero_grad()
                outputs = model(sentences)
            elif model_choice == 'c':
                word_inputs, prefix_inputs, suffix_inputs, tags = batch
                optimizer.zero_grad()
                outputs = model(word_inputs, prefix_inputs, suffix_inputs)
            elif model_choice == 'd':
                sentences_char, sentences_word, tags = batch
                optimizer.zero_grad()
                outputs = model(sentences_char, sentences_word)

            outputs = outputs.view(-1, tagset_size)
            tags = tags.view(-1)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()

            processed_sentences += len(batch[0])

            if processed_sentences >= eval_interval:
                processed_sentences = 0
                model.eval()
                total_loss = 0
                total_accuracy = 0
                with torch.no_grad():
                    for dev_batch in dev_loader:
                        if model_choice in ['a', 'b']:
                            dev_sentences, dev_tags = dev_batch
                            dev_outputs = model(dev_sentences)
                        elif model_choice == 'c':
                            dev_word_inputs, dev_prefix_inputs, dev_suffix_inputs, dev_tags = dev_batch
                            dev_outputs = model(dev_word_inputs, dev_prefix_inputs, dev_suffix_inputs)
                        elif model_choice == 'd':
                            dev_sentences_char, dev_sentences_word, dev_tags = dev_batch
                            dev_outputs = model(dev_sentences_char, dev_sentences_word)

                        dev_outputs = dev_outputs.view(-1, tagset_size)
                        dev_tags = dev_tags.view(-1)
                        dev_loss = criterion(dev_outputs, dev_tags)
                        total_loss += dev_loss.item()
                        total_accuracy += calculate_accuracy(dev_outputs, dev_tags, tagset_size)
                avg_loss = total_loss / len(dev_loader)
                avg_accuracy = total_accuracy / len(dev_loader)
                print(f"Epoch {epoch + 1}/{epochs}, Eval Loss: {avg_loss}, Eval Accuracy: {avg_accuracy}")

        print(f"Epoch {epoch + 1}/{epochs} completed")

    # Save the trained model and additional data
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'tag_set': tag_set,
        'char_vocab': char_vocab if model_choice == 'b' or model_choice == 'd' else None,
        'embedding_matrix': embedding_matrix if model_choice == 'c' else None,
        'prefix_vocab': prefix_vocab if model_choice == 'c' else None,
        'suffix_vocab': suffix_vocab if model_choice == 'c' else None
    }, model_file)
    print(f"Model saved to {model_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('repr', choices=['a', 'b', 'c', 'd'], help="Model representation: a, b, c, or d")
    parser.add_argument('trainFile', help="Input file to train on")
    parser.add_argument('modelFile', help="File to save the trained model")
    parser.add_argument('--devFile', default='pos/dev', help="Development file for evaluation")
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs for training")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    parser.add_argument('--eval_interval', type=int, default=500,
                        help="Number of sentences to process before evaluation")
    args = parser.parse_args()

    main_train(args.repr, args.trainFile, args.modelFile, args.devFile, args.epochs, args.batch_size,
               args.eval_interval)
