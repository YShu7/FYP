import torch
import torch.nn as nn

DEBUG = False

class Net(nn.Module):

  def __init__(self, hidden_dim, num_layers=1, bidirectional=True):
    super(Net, self).__init__()
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

def evaluate(model, loader, criterion):
    model.train(False)

    running_loss = 0
    for data in loader:
        inputs, targets = data
        outputs = model(inputs.type(torch.FloatTensor).to(device)).to(device).reshape(-1)  # forward pass
        loss = criterion(outputs, targets.type(torch.FloatTensor).to(device))
        running_loss += loss.item()
    epoch_loss = running_loss / len(loader)

    return epoch_loss
