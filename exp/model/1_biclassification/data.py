import collections
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
        
        inputs = torch.tensor(self.inputs[idx])
        targets = torch.tensor(self.targets[idx], dtype=torch.long)
        sample = (inputs, targets)
        if self.transform:
            sample = self.transform(sample)

        return sample

def prepare_data(df_contacts, to_ix, input_dim):
    start_time_to_idx = collections.defaultdict(set)
    for _, row in df_contacts.iterrows():
        start_time_to_idx[row['time']].add(row['agent_id'])

    real_ids = set(df_contacts['real_id'].tolist())
    input, candidate, target = [], [], []

    neg, pos = 0, 0

    for real_id in real_ids:
        walk = df_contacts.loc[df_contacts['real_id'] == real_id].sort_values('time')
        path, times = [], []

        for i, row in walk.iterrows():
            times.append(int(row['time']))
            path.append(to_ix[row['agent_id']])

        start_idx = 0
        while True:
            try:
                curr_input = path[start_idx:start_idx+input_dim]
                curr_candidates = [to_ix[idx] for idx in start_time_to_idx[times[start_idx+input_dim]]]
                curr_target = path[start_idx+input_dim]
                start_idx += input_dim

                num_neg = 0
                for curr_candidate in curr_candidates:
                    if curr_target == curr_candidate:
                        input.append(curr_input + [curr_candidate])
                        pos += 1
                        target.append(1)
                    elif curr_target != curr_candidate and num_neg < 1:
                        input.append(curr_input + [curr_candidate])
                        neg += 1
                        num_neg += 1
                        target.append(0)
            except:
                break

    print(len(input), len(target), f'{pos}:{neg}')

    return input, target