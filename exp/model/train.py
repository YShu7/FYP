import argparse
import sqlite3
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

REGRESSION = 0
BI_CLASS = 1
MULTI_CLASS = 2

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=int, default=0,
                    help="""
                    {}: Regressiont \n
                    {}: Binary Classification \n
                    {}: Multiple Classification
                    """.format(REGRESSION, BI_CLASS, MULTI_CLASS))
parser.add_argument('--bidirectional', action='store_true')
parser.set_defaults(bidirectional=False)

args = parser.parse_args()

if args.type == REGRESSION:
    from regression.data import prepare_data, PathDataset
    from regression.net import Net, evaluate
elif args.type == BI_CLASS:
    from biclassification.data import prepare_data, PathDataset
    from biclassification.net import Net, evaluate
elif args.type == MULTI_CLASS:
    pass

torch.manual_seed(1)
torch.use_deterministic_algorithms(True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INTERVAL = 60
TIME_PERIOD = 1
MINUTE = 10
UPDATE_TIME = MINUTE * INTERVAL
N, M = 30, 91
PATH = "../../corona-sniffer/backend/data"
TRAIN_DB = 'database_100+_train_2.db'
TEST_DB = 'database_100-_test_2.db'

EMBEDDING_DIM = 32
HIDDEN_DIM = 128
INPUT_DIM = 10

cnx = sqlite3.connect(f'{PATH}/{TRAIN_DB}')
df_agents = pd.read_sql_query("SELECT * FROM agents", cnx)
df_contacts_tmp = pd.read_sql_query("SELECT * FROM contacts JOIN walkers ON walkers.id = contacts.walker_id", cnx)
idx = df_contacts_tmp.groupby(['walker_id', 'time'])['distance'].transform(min) == df_contacts_tmp['distance']
df_contacts = df_contacts_tmp[idx]

agent_to_ix = {}
for _, row in df_agents.iterrows():
  x, y = row['id'].split('EVEN')
  agent_to_ix[row['id']] = [int(x)/N, (int(y)%M)/M]

input_train, target_train = prepare_data(df_contacts, agent_to_ix, INPUT_DIM)
train_loader = DataLoader(PathDataset(input_train[100:], target_train[100:], len(agent_to_ix)), batch_size=64, shuffle =True)
val_loader = DataLoader(PathDataset(input_train[:100], target_train[:100], len(agent_to_ix)), batch_size=1, shuffle =False)

def train(model, train_loader, val_loader, criterion):
    for epoch in range(1000):  # again, normally you would NOT do 300 epochs, it is toy data
        model.train(True)
        for data in train_loader:
            inputs, targets = data
            outputs = model(inputs.type(torch.FloatTensor).to(device)).to(device).reshape(-1)  # forward pass
            model.zero_grad()
            optimizer.zero_grad()  # caluclate the gradient, manually setting to 0
            loss = criterion(outputs, targets.type(torch.FloatTensor).to(device))  # obtain the loss function
            loss.backward()  # calculates the loss of the loss function
            optimizer.step()  # improve from loss, i.e backprop

        if (epoch + 1) % 50 == 0:
            print("Epoch: %d" % (epoch + 1))
            epoch_loss = evaluate(model, val_loader, criterion)

            torch.save(model.state_dict(), f'{args.type}_e{epoch + 1}.pkl')

model = Net(HIDDEN_DIM, bidirectional=args.bidirectional).to(device)
print(model)
criterion = torch.nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

train(model, train_loader, val_loader, criterion)