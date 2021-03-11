import torch
import torch.nn as nn


class Classifier(nn.Module):

	def __init__(self, vocab_size, embedd_dim, n_layers=1, embedd_vec=None, 
		drop_val=0.25, hidden_dim=128, is_trainable=True, bidirectional=True, device=None):

		super(Classifier, self).__init__()

		self.vocab_size = vocab_size
		self.embedd_dim = embedd_dim
		self.n_layers = n_layers
		self.hidden_dim = hidden_dim
		self.num_directions = 2 if bidirectional else 1

		self.device = device

		self.embedd = nn.Embedding(self.vocab_size, self.embedd_dim)

		if embedd_vec is not None:
			self.embedd.weight.data.copy_(embedd_vec)

		if not is_trainable:
			self.embedd.weight.requires_grad = False



		self.lstm = nn.LSTM(self.embedd_dim, self.hidden_dim, num_layers=self.n_layers, batch_first=True, bidirectional=bidirectional)
		self.linear = nn.Linear(self.num_directions * self.hidden_dim, 1)
		self.drop = nn.Dropout(p=drop_val)


	def forward(self, X, X_len):

		h0 = torch.zeros(self.n_layers * self.num_directions, X.size(0), self.hidden_dim).to(self.device)
		c0 = torch.zeros(self.n_layers * self.num_directions, X.size(0), self.hidden_dim).to(self.device)

		# Pass through embedding layer
		X = self.embedd(X)

		lstm_out, (h, c) = self.lstm(X, (h0, c0))

		out, _ = torch.max(lstm_out, 1)

		out = self.drop(out)
		out = self.linear(out)

		return out


