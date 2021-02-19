# python3.5 buildtagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(1)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
init_time = datetime.datetime.now()


class LSTMTagger(nn.Module):
    def __init__(self, word_embed_dim, char_embde_dim, hidden_dim, word_to_idx, char_to_idx, tag_to_idx, n_layers,
                 dropout):
        super(LSTMTagger, self).__init__()
        word_to_idx_size, char_to_idx_size, tag_vocab_size = len(word_to_idx) + 1, len(char_to_idx) + 1, len(tag_to_idx)
        self.word_to_idx, self.char_to_idx, self.tag_to_idx = word_to_idx, char_to_idx, tag_to_idx
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(word_to_idx_size, word_embed_dim).to(device)
        self.char_embeddings = nn.Embedding(char_to_idx_size, char_embde_dim).to(device)

        self.conv1d = nn.Conv1d(in_channels=char_embde_dim, out_channels=char_embde_dim, kernel_size=3,
                                stride=1, padding=1, bias=True).to(device)
        self.pool = nn.AdaptiveMaxPool1d(1).to(device)

        self.lstm = nn.LSTM(input_size=word_embed_dim + char_embde_dim, hidden_size=hidden_dim,
                            num_layers=n_layers, dropout=dropout if n_layers > 1 else 0, bidirectional=True).to(device)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, tag_vocab_size).to(device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, words):
        word_idxs = torch.tensor([self.word_to_idx[word] if word in self.word_to_idx else 0 for word in words]).to(
            device)
        word_embeds = self.word_embeddings(word_idxs).unsqueeze(1).to(device)

        char_idxs = nn.utils.rnn.pad_sequence(
            [torch.tensor([self.char_to_idx[c] if c in self.char_to_idx else 0 for c in word]).to(device) for word in
             words],
            batch_first=True)
        char_embeds = self.char_embeddings(char_idxs).transpose(1, 2).to(device)

        char_rep = self.conv1d(char_embeds).to(device)
        char_rep = self.pool(F.relu(char_rep)).transpose(1, 2).to(device)
        print("F", word_idxs.shape, word_embeds.shape, char_idxs.shape, char_embeds.shape, char_rep.shape)
        word_rep = torch.cat((word_embeds, char_rep), dim=2).to(device)
        print(word_rep.shape)
        self.lstm.flatten_parameters()
        output, _ = self.lstm(word_rep)
        print(output.shape)
        out = self.hidden2tag(output.view(len(words), -1))
        print(out.shape)
        out = F.log_softmax(out, dim=1)
        return out


class LSTMDataset(Dataset):
    def __init__(self, trainind_data):
        self.trainind_data = trainind_data

    def __len__(self):
        return len(self.trainind_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.trainind_data[idx]


def get_idx_dic(train_file):
    with open(train_file, 'r') as f:
        lines = f.readlines()

    training_data = []
    word_to_idx, char_to_idx, tag_to_idx, idx_to_tag = {}, {}, {}, {}
    curr_word_idx, curr_char_idx, curr_tag_idx = 1, 1, 0

    for line in lines:
        tokens = line.strip().split(' ')
        words = []
        tags = []

        for token in tokens:
            word, tag = token.rsplit('/', 1)

            if word not in word_to_idx:
                word_to_idx[word] = curr_word_idx
                curr_word_idx += 1

            if tag not in tag_to_idx:
                tag_to_idx[tag] = curr_tag_idx
                idx_to_tag[curr_tag_idx] = tag
                curr_tag_idx += 1

            for char in word:
                if char in char_to_idx:
                    continue
                char_to_idx[char] = curr_char_idx
                curr_char_idx += 1
            words.append(word)
            tags.append(tag_to_idx[tag])

        training_data.append((words, tags))
    return char_to_idx, word_to_idx, tag_to_idx, idx_to_tag, training_data


def train_model(train_file, model_file):
    char_to_idx, word_to_idx, tag_to_idx, idx_to_tag, training_data = get_idx_dic(train_file)

    dataset = LSTMDataset(training_data)
    losses = []
    loss_function = nn.CrossEntropyLoss()
    model = LSTMTagger(128, 32, 32, word_to_idx, char_to_idx, tag_to_idx, 2, 0.1)
    print(model)
    optimizer = optim.Adam(params=model.parameters(), lr=0.002)

    epoches = 100
    size = int(len(training_data) / epoches + 1)
    random.shuffle(training_data)
    for epoch in range(epoches):
        epoch_loss = 0

        for i, (words, tags_idx) in enumerate(dataset[size * epoch:min(size * (epoch + 1), len(training_data))]):
            model.zero_grad()
            tag_scores = model(words).to(device)
            loss = loss_function(tag_scores, torch.tensor(tags_idx).to(device)).to(device)
            print("tag", tag_scores.shape, torch.tensor(tags_idx).shape)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            time_diff = datetime.datetime.now() - init_time

            if time_diff > datetime.timedelta(minutes=9, seconds=45):
                torch.save((word_to_idx, char_to_idx, tag_to_idx, idx_to_tag, model.state_dict()), model_file)
                return
            print(loss.item())

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Time Elapsed: {}'
              .format(epoch + 1, epoches, i + 1,
                      len(training_data[size * epoch:min(size * (epoch + 1), len(training_data))]), epoch_loss,
                      time_diff))

        losses.append(epoch_loss)
        print(sum(losses) / len(losses))
        torch.save((word_to_idx, char_to_idx, tag_to_idx, model.state_dict()), model_file)
    print('Finished...')


if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
