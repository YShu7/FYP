import torch
from torch.utils.data import Dataset


class PathDataset(Dataset):

    def __init__(self, inputs, targets, output_size, transform=None):
        self.inputs = inputs
        self.targets = targets
        self.output_size = output_size
        self.transform = transform

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        inputs = torch.tensor(self.inputs[idx], dtype=torch.float)
        targets = torch.tensor(self.targets[idx], dtype=torch.float)
        sample = (inputs, targets)
        if self.transform:
            sample = self.transform(sample)

        return sample


def prepare_data(df_contacts, to_ix, input_dim):
  real_ids = set(df_contacts['real_id'].tolist())
  input, target = [], []

  for real_id in real_ids:
    walk = df_contacts.loc[df_contacts['real_id'] == real_id].sort_values('time')
    path, times = [], []

    for i, row in walk.iterrows():
      times.append(int(row['time']))
      path.append(to_ix[row['agent_id']])

    start_idx = 0
    while True:
      try:
        curr_input = path[start_idx:start_idx+10]
        curr_target = path[start_idx+10]
        start_idx += input_dim

        input.append(curr_input)
        target.append(curr_target)
      except:
        break

  print(len(input), len(target))
  return input, target