import os
import glob
import deepdish as dd
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset

class ADNIDataset(InMemoryDataset):
    def __init__(self, root, name='ADNI', transform=None, pre_transform=None):
        self.name = name
        super(ADNIDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        raw_files = glob.glob(os.path.join(self.root, 'raw', '*.h5'))
        for h5_file in raw_files:
            try:
                sample = dd.io.load(h5_file)
                conn = sample['corr']  # (nroi, nroi)
                label = int(sample['label'])  # 0,1,2

                # Replace infs or NaNs
                conn[np.isinf(conn)] = 0
                conn[np.isnan(conn)] = 0

                # Convert to graph
                edge_index = np.array(np.nonzero(conn))
                edge_weight = conn[edge_index[0], edge_index[1]]
                x = torch.tensor(conn, dtype=torch.float32)  # can also use diag or stats
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                edge_attr = torch.tensor(edge_weight, dtype=torch.float32)

                data = Data(
                    x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=torch.tensor([label], dtype=torch.long),
                    pos=None  # you can set to coordinates or leave None
                )
                data_list.append(data)
            except Exception as e:
                print(f"Failed to load {h5_file}: {e}")



        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])