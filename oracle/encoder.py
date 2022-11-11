import os
import json
import numpy as np
import torch
import torch.nn as nn

from typing import Tuple, List
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset, DataLoader


class DatasetCent(Dataset):

	# Data: Pass X and Y as a tuple i.e., data = (X, Y)
	def __init__(self, data):
		super().__init__()

		x = data[0]
		y = data[1]

		self.x = torch.from_numpy(x)
		self.y = torch.from_numpy(y)
		self.n_samples = x.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.n_samples

    
class EconderArchitecture(nn.Module):

	def __init__(self, input_size, hidden_layers=3, output_layer="linear"):

		super().__init__()

		self.input_size = input_size
		self.output_size = input_size
		self.encoder = nn.ModuleList()

		for i in range(hidden_layers):
			self.encoder.append(nn.Linear(self.input_size, self.output_size))
			self.encoder.append(nn.ReLU())

		self.encoder.append(nn.Linear(self.input_size, self.output_size))
		self.encoder.append(nn.Sigmoid())


	def forward(self, x):
		for layer in self.encoder:
			x = layer(x)
		return x

class Encoder:

	def __init__(self, input_size, hidden_layers=1):

		self.input_size = input_size
		self.device = torch.device(
			'cuda' if torch.cuda.is_available() else 'cpu')
		self.model = EconderArchitecture(
			self.input_size, hidden_layers=hidden_layers).to(self.device)

	def fit(self, data: DatasetCent, epochs=80, batch_size=64):

		criterion = nn.HuberLoss()
		optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

		train_loader = DataLoader(
			dataset=data, batch_size=batch_size, shuffle=True)

		# Train the model
		n_total_steps = len(train_loader)
		for epoch in range(epochs):
			for i, (x, y) in enumerate(train_loader):

				x = x.to(self.device)
				y = y.to(self.device)

				# Forward pass
				outputs = self.model(x.float())
				loss = criterion(outputs.float(), y.float())

				# Backward and optimize
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				if (i+1) % 100 == 0:
					print(
						f'\t\tEpoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}', end="\r")
		print("\n")

	def predict(self, X: torch.Tensor):

		with torch.no_grad():
			X_pred = []
			for x in X.to(self.device):
				pred = self.model(x.float())
				X_pred.append(pred)

		return np.array(torch.stack(X_pred).cpu().numpy())


class EncoderMFs:

	def transform(self, X_train, y_train, X_test, params):

		data_loader = DatasetCent((X_train, y_train))
		enc = Encoder(X_train.shape[1], params["hidden_layers"])
		enc.fit(data_loader)
		try:
			return enc.predict(torch.from_numpy(X_test.todense()))
		except:
			return enc.predict(torch.from_numpy(X_test))

def load_x_y(
        file: str,
        test_train: str
) -> Tuple[np.ndarray, np.ndarray]:

    loaded = np.load(file, allow_pickle=True)
    X = loaded[f"X_{test_train}"]
    
    if f"y_{test_train}" not in loaded:
        return X, None

    y = loaded[f"y_{test_train}"]

    if X.size == 1:
        X = X.item()

    return X, y

def read_train_test_meta_oracle(
        dir_meta_input: str,
        dataset: str,
        n_folds: int,
        fold_id: int,
        algorithms: List[str],
        oracle_path: str,
        oracle_strategy: str
) -> Tuple[np.ndarray, np.ndarray]:
    Xs_train, Xs_test = [], []

    for alg in algorithms:
        file_train_meta = f"{dir_meta_input}/{dataset}/{n_folds}_folds/{alg}/{fold_id}/train.npz"
        file_test_meta = f"{dir_meta_input}/{dataset}/{n_folds}_folds/{alg}/{fold_id}/test.npz"

        X_train_meta, _ = load_x_y(file_train_meta, 'train')
        X_test_meta, _ = load_x_y(file_test_meta, 'test')

        if oracle_strategy in ["upper_bound", "upper_test", "upper_train"]:
            oracle_base_dir = f"{oracle_path}/{oracle_strategy}/{dataset}/{alg}/{fold_id}"
            oracle_file_train = f"{oracle_base_dir}/train.npz"
            oracle_file_test = f"{oracle_base_dir}/test.npz"

            oracle_train = np.load(oracle_file_train)['y']
            oracle_test = np.load(oracle_file_test)['y']

            Xs_train.append(X_train_meta * oracle_train[:, None])
            Xs_test.append(X_test_meta * oracle_test[:, None])
        else:
            Xs_train.append(X_train_meta)
            Xs_test.append(X_test_meta)

    X_train_meta = np.hstack(Xs_train)
    X_test_meta = np.hstack(Xs_test)

    return X_train_meta, X_test_meta


DATA_SOURCE = "/home/welton/data"
META_INPUT = f"{DATA_SOURCE}/clfs_output/split_10"
ORACLE_INPUT = f"{DATA_SOURCE}/oracle"
DATASETS = ["webkb"]
CLFS = ["kpr", "ktr", "lpr", "ltr", "sfr", "stmk", "xfr", "xpr", "xtr", "kfr", "ktmk", "lfr", "ltmk", "spr", "str", "xlnet_softmax", "xtmk", "rep_bert"]


for dataset in DATASETS:
    print(f"{dataset.upper()}")
    for fold in np.arange(10):
        print(f"\tFOLD: {fold}")
        # Loading original probabilities.
        X_train, X_test = read_train_test_meta_oracle(META_INPUT, dataset, 10, fold, CLFS, ORACLE_INPUT, "normal")
        # Loading upper_boung probabilities.
        OX_train, OX_test = read_train_test_meta_oracle(META_INPUT, dataset, 10, fold, CLFS, ORACLE_INPUT, "upper_bound")
        
        # Training the encoder.
        data_loader = DatasetCent((X_train, OX_train))
        encoder = Encoder(X_train.shape[1])
        encoder.fit(data_loader)
        pred = encoder.predict(torch.from_numpy(X_test))

        output_dir = f"{ORACLE_INPUT}/encoder/{dataset}/all_clfs/{fold}"
        os.makedirs(output_dir, exist_ok=True)

        np.savez(f"{output_dir}/test", X_test=pred)
        np.savez(f"{output_dir}/train", X_train=X_train)