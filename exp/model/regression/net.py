import torch
import torch.nn as nn

class Net(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, num_classes, num_layers=1, seq_len=10):
    super(Net, self).__init__()
    self.hidden_dim = hidden_dim
    self.num_layers = num_layers
    self.embedding_dim = embedding_dim


    # LSTM
    self.lstm = nn.LSTM(2, hidden_dim, num_layers=num_layers, batch_first=True) #lstm
    self.lstm_1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
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
    hn = hn.view(-1, self.hidden_dim)
    if DEBUG: print('hn:', hn.shape)

    out = self.relu(self.fc_1(hn))
    if DEBUG: print('fc_1:', out.shape)

    out = self.fc(out)
    return out

def evaluate(model, loader, criterion):
  model.train(False)

  running_loss = 0
  running_corrects = 0

  for data in loader:

    inputs, targets = data
    outputs = model(inputs)
    loss = criterion(outputs, targets)

    running_loss += loss.item()

    for pred, target in zip(outputs, targets):
      x_pred, x_target = round(float(pred[0])*N), round(float(target[0])*N)
      y_pred = round(float(pred[1])*M) + x_pred*M
      y_target = round(float(target[1]*M)) + x_target*M
      if x_pred == x_target and \
        y_pred == y_target:
        running_corrects += 1
  
  epoch_loss = running_loss / len(loader)
  epochs_acc = running_corrects / len(loader) * 100

  return epoch_loss, epochs_acc