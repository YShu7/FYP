import torch
import torch.nn as nn
import math

DEBUG = False

class Net(nn.Module):

  def __init__(self, hidden_dim, num_layers=1, bidirectional=True):
    super(Net, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional

    if bidirectional:
      self.lstm = nn.LSTM(2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)  # lstm
      self.lstm_1 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)
      self.fc_1 = nn.Linear(hidden_dim*2, hidden_dim)
    else:
      self.lstm = nn.LSTM(2, hidden_dim, num_layers=num_layers, bidirectional=False, batch_first=True) #lstm
      self.lstm_1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=False, batch_first=True)
      self.fc_1 = nn.Linear(hidden_dim, hidden_dim)

    self.fc = nn.Linear(hidden_dim, 2)
    self.relu = nn.ReLU() 

  def forward(self, input):
    # INPUT: (batch, seq_len)
    if DEBUG: print('input: ', input.shape)

    # LSTM
    # LSTM: (batch, seq, feature)
    lstm, (hn, cn) = self.lstm(input)
    lstm = nn.Dropout(0.2)(lstm)
    if DEBUG: print('lstm:', lstm.shape)

    lstm_1, (hn, cn) = self.lstm_1(lstm)
    hn = nn.Dropout(0.2)(hn)
    if self.bidirectional:
      hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
    else:
      hn = hn.view(-1, self.hidden_dim)
    if DEBUG: print('hn:', hn.shape)

    out = self.relu(self.fc_1(hn))
    if DEBUG: print('fc_1:', out.shape)

    out = self.fc(out)
    return out

  def prepare_data_test(self, walk, to_ix):
    path = []

    for i, row in walk.iterrows():
      path.append(to_ix[row['agent_id']])

    return torch.Tensor([path]).type(torch.FloatTensor)

  def compute_proba(self, walk, candidates, to_ix, N, M, INTERVAL):
    candidate_agent_probs = {}

    output = self(self.prepare_data_test(walk, to_ix))[0]  # forward pass
    output = [round(float(output[0]) * N), round(float(output[1]) * M)]
    for _, candidate in candidates.iterrows():
      x, y = to_ix[candidate['agent_id']]
      prob = 1 - 1 / math.sqrt(N ** 2 + M ** 2) * math.sqrt((int(x) - output[0]) ** 2 + (int(y) - output[1]) ** 2)
      candidate_agent_probs[candidate['walker_id']] = prob

    sum_prob = sum(candidate_agent_probs.values())
    for k, v in candidate_agent_probs.items():
      candidate_agent_probs[k] = candidate_agent_probs[k] / sum_prob

    return candidate_agent_probs

  def evaluate(self, loader, criterion, device):
    self.train(False)
    running_loss = 0

    for data in loader:
      inputs, targets = data
      outputs = self(inputs.to(device))
      loss = criterion(outputs.to(device), targets.to(device))
      running_loss += loss.item()

    epoch_loss = running_loss / len(loader)
    print("loss: %1.5f" % (epoch_loss))
    return epoch_loss
