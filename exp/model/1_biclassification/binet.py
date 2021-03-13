import torch
import torch.nn as nn

DEBUG = False

class BiNet(nn.Module):

  def __init__(self, hidden_dim, num_layers=1, bidirectional=True):
    super(BiNet, self).__init__()
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

    self.fc = nn.Linear(hidden_dim, 1)
    self.relu = nn.ReLU()


  def forward(self, input):
    # LSTM
    lstm, (hn, cn) = self.lstm(input)
    lstm = nn.Dropout(0.2)(lstm)
    if DEBUG: print('lstm (batch, seq, feature):', lstm.shape)

    lstm_1, (hn, cn) = self.lstm_1(lstm)
    hn = nn.Dropout(0.2)(hn)
    if self.bidirectional:
        hn = torch.cat((hn[-2,:,:], hn[-1,:,:]), dim = 1)
    else:
        hn = hn.view(-1, self.hidden_dim)
    if DEBUG: print('hn (batch, 1, feature):', hn.shape)

    out = self.relu(self.fc_1(hn)) #first Dense
    if DEBUG: print('fc_1:', out.shape)

    out = self.fc(out)
    if DEBUG: print('fc:', out.shape)

    return out

  def prepare_data_test(self, walk, candidate, num_steps, to_ix):
      path = []

      for i, row in walk.iterrows():
          path.append(to_ix[row['agent_id']])

      last_row = walk.iloc[-1]['agent_id']
      for i in range(num_steps):
          path.append([
              to_ix[last_row][0] + i * (to_ix[candidate][0] - to_ix[last_row][0]) / num_steps,
              to_ix[last_row][1] + i * (to_ix[candidate][1] - to_ix[last_row][1]) / num_steps
          ])
      path.append(to_ix[candidate])
      return torch.Tensor([path]).type(torch.FloatTensor)

  def compute_proba(self, walk, candidates, to_ix, N, M, INTERVAL):
      candidate_agent_probs = {}

      for _, candidate in candidates.iterrows():
          num_steps = int((candidate['time'] - walk.iloc[-1]['time']) / INTERVAL) - 1
          candidate_agent_probs[candidate['walker_id']] = \
          torch.sigmoid(self(self.prepare_data_test(walk, candidate['agent_id'], num_steps, to_ix)))[0]

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
          outputs = self(inputs.type(torch.FloatTensor).to(device)).to(device).reshape(-1)  # forward pass
          preds = torch.round(torch.sigmoid(outputs)).type(torch.LongTensor)

          loss = criterion(outputs, targets.type(torch.FloatTensor).to(device))

          running_loss += loss.item()
          running_corrects += torch.sum(preds.to(device) == targets.to(device)).item()

      epoch_loss = running_loss / len(loader)
      epochs_acc = running_corrects / len(loader)
      print("loss: %1.5f, acc: %1.5f" % (epoch_loss, epochs_acc))
      return epoch_loss, epochs_acc
