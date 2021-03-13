import torch
import torch.nn as nn

DEBUG = False

class MultiNet(nn.Module):

  def __init__(self, hidden_dim, num_layers=1, bidirectional=True):
    super(MultiNet, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.bidirectional = bidirectional

    if bidirectional:
      self.lstm = nn.LSTM(2, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True) #lstm
      self.lstm_1 = nn.LSTM(hidden_dim*2, hidden_dim, bidirectional=True, batch_first=True)
      self.fc_1 = nn.Linear(hidden_dim*2, hidden_dim)
    else:
      self.lstm = nn.LSTM(2, hidden_dim, num_layers=num_layers, bidirectional=False, batch_first=True)  # lstm
      self.lstm_1 = nn.LSTM(hidden_dim, hidden_dim, bidirectional=False, batch_first=True)
      self.fc_1 = nn.Linear(hidden_dim, hidden_dim)

    self.fc = nn.Linear(hidden_dim, 30*91)
    self.relu = nn.ReLU()


  def forward(self, input):

    lstm, (hn, cn) = self.lstm(input)
    lstm = nn.Dropout(0.2)(lstm)
    if DEBUG: print('lstm (batch, seq, feature):', lstm.shape)

    lstm_1, (hn, cn) = self.lstm_1(lstm)
    hn = nn.Dropout(0.2)(hn)
    if self.bidirectional:
      hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
    else:
      hn = hn.view(-1, self.hidden_dim)
    if DEBUG: print('hn (batch, 1, feature):', hn.shape)

    out = self.relu(self.fc_1(hn))
    if DEBUG: print('fc_1:', out.shape)

    out = self.fc(out)
    if DEBUG: print('fc:', out.shape)

    return out

  def prepare_data_test(self, walk, to_ix):
    path = []

    for i, row in walk.iterrows():
      path.append(to_ix[row['agent_id']])
    return torch.Tensor([path]).type(torch.FloatTensor)

  def compute_proba(self, walk, candidates, to_ix, N, M, INTERVAL):
    candidate_agent_probs = {}

    outputs = []
    next_walk = self.prepare_data_test(walk, to_ix)
    outputs.append(torch.softmax(self(next_walk), 1)[0])
    next_walk = torch.cat([next_walk,
                           torch.Tensor([[[int(torch.max(outputs[-1]) / M) / N,
                                           torch.max(outputs[-1]) % M / M]]])], 1)
    outputs.append(torch.softmax(self(next_walk), 1)[0])
    next_walk = torch.cat([next_walk,
                           torch.Tensor([[[int(torch.max(outputs[-1]) / M) / N,
                                           torch.max(outputs[-1]) % M / M]]])], 1)
    outputs.append(torch.softmax(self(next_walk), 1)[0])
    for _, candidate in candidates.iterrows():
      num_steps = int((candidate['time'] - walk.iloc[-1]['time']) / INTERVAL) - 1
      class_num = int(candidate['agent_id'].split('EVEN')[1])
      candidate_agent_probs[candidate['walker_id']] = outputs[num_steps][class_num]

    sum_prob = sum(candidate_agent_probs.values())
    for k, v in candidate_agent_probs.items():
      candidate_agent_probs[k] = candidate_agent_probs[k] / sum_prob

    return candidate_agent_probs

  def evaluate(self, loader, criterion, device):
    self.train(False)

    running_loss = 0
    running_corrects = 0

    for data in loader:
      inputs, targets = data
      outputs = self(inputs.to(device))
      _, preds = torch.max(torch.softmax(outputs.to(device), 1), 1)

      loss = criterion(outputs.to(device), targets.to(device))

      running_loss += loss.item()
      running_corrects += torch.sum(preds.to(device) == targets.to(device)).item()

    epoch_loss = running_loss / len(loader)
    epochs_acc = running_corrects / len(loader)
    print("loss: %1.5f, acc: %1.5f" % (epoch_loss, epochs_acc))
    return epoch_loss, epochs_acc
