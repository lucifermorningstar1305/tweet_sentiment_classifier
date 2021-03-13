import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchtext
import torchtext.legacy.data as tld
import os
import sys
import time
import argparse
import wandb

from model import Classifier
from train import train
from utils import *



def setup_train(epochs, lr, tbz, vbz, seed, is_wandb=False, is_plot=True):

	# Define the TEXT and LABEL Field
	TEXT = tld.Field(sequential=True, batch_first=True, lower=False, tokenize=str.split, fix_length=120, pad_first=True, include_lengths=True)
	LABEL = tld.Field(sequential=False, use_vocab=False, is_target=True, dtype=torch.float)

	# Create the TabularDataset
	train_dataset = tld.TabularDataset(path="../DATA/train.csv", format="csv", skip_header=True, fields=[('target', LABEL), ('text', TEXT)])
	valid_dataset = tld.TabularDataset(path="../DATA/valid.csv", format="csv", skip_header=True, fields=[('target', LABEL), ('text', TEXT)])

	# Build the vocabulary
	TEXT.build_vocab(train_dataset, min_freq=5, vectors="glove.6B.100d")

	vocab = TEXT.vocab

	vocab_vec = vocab.vectors


	vocab_df = pd.DataFrame({"words" : list(vocab.stoi.keys()), "index":list(vocab.stoi.values())})
	vocab_df.to_csv("../DATA/vocab.csv", index=False)

	print(f'Vocab Length : {len(vocab)}')

	# Build the iterator
	train_iter = tld.Iterator(train_dataset, batch_size=tbz, sort_key=lambda x:len(x.text), train=True, shuffle=True)
	valid_iter = tld.Iterator(valid_dataset, batch_size=vbz, sort_key=lambda x: len(x.text), train=False, shuffle=False)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = Classifier(len(vocab), vocab_vec.size(1), hidden_dim=128, embedd_vec=vocab_vec, device=device, is_trainable=False)

	for layer in model.lstm.named_parameters():

		if 'weight' in layer[0]:
			# torch.nn.init.orthogonal_(layer[1])
			torch.nn.init.xavier_normal_(layer[1])

	torch.nn.init.xavier_normal_(model.linear.weight)

	if is_wandb:
		wandb.watch(model, log='all')

	model = model.to(device)

	optimizer = torch.optim.Adam(model.parameters(), lr=lr)

	torch.manual_seed(seed)

	train_losses, valid_losses, valid_acc, valid_f1 = train(model, optimizer, train_iter, valid_iter,
	 epochs=epochs, device=device, islog=is_wandb, do_clip = False, file_path="../MODELS/")

	if is_plot:

		plt.title("Epochs vs Losses")
		plt.plot(train_losses, label='Training loss')
		plt.plot(valid_losses, label='Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Losses')
		plt.legend()
		plt.show()

		plt.title("Epochs vs Acc")
		plt.plot(valid_acc)
		plt.xlabel("Epochs")
		plt.ylabel("Acc")
		plt.show()

		plt.title("F1-Score vs Acc")
		plt.plot(valid_f1)
		plt.xlabel("Epochs")
		plt.ylabel("F1-Score")
		plt.show()
		



def setup_test():
	
	text = input("Enter a text : ")
	text = preprocess(text)
	text = text.split()

	text = pad_sequence(text)

	vocab_df = pd.read_csv("../DATA/vocab.csv")
	vocab = {k:v for k, v in zip(vocab_df['words'].values, vocab_df['index'].values)}

	encoded_sent = encode_sentence(vocab, text)

	embedd_mat = None

	if not os.path.exists("../MODELS/glove100d.pt"):

		glove_vec = load_glove_vectors("./.vector_cache/glove.6B.100d.txt")
		embedd_mat = get_glove_vectors(glove_vec, vocab, embedd_size=100)
		torch.save(embedd_mat, "../MODELS/glove100d.pt")

	else:
		embedd_mat = torch.load("../MODELS/glove100d.pt", map_location=torch.device("cpu"))

	device = torch.device("cpu")
	model = Classifier(len(vocab), 100, hidden_dim=128, embedd_vec=embedd_mat, device=device, drop_val=0.4, is_trainable=False)
	model = load_chkpt(model, file_path="../MODELS/model.pt").to(device)
	model.eval()
	print("Model Load complete...")

	encoded_sent = encoded_sent.unsqueeze(0)

	y_pred = model(encoded_sent, len(encoded_sent))
	print(y_pred)
	y_pred = (y_pred > 0)
	y_pred = y_pred.detach().numpy()
	y_pred = int(y_pred.reshape(-1)[0])

	prediction = {0:"Negative", 1:"Positive"}
	







if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--action', '-a', help='whether to train/test the model')
	parser.add_argument('--epochs', '-e', type=int, default=20, help='define the number of epochs')
	parser.add_argument('--learning', '-lr', type=float, default=0.001, help='define the learning rate of the model')
	parser.add_argument('--train_batch_sz', '-tbz', type=int, default=32, help='define the training batch size')
	parser.add_argument('--valid_batch_sz', '-vbz', type=int, default=256, help='define the validation batch size')
	parser.add_argument('--wandb', '-w', type=int, default=0, help='decide whether to use wandb for logging 0/1')
	parser.add_argument('--seed', '-s', type=int, default=42, help='define the seed value for pytorch random state')

	args = parser.parse_args()


	config = {
		'action' : args.action,
		'epochs' : args.epochs,
		'learning' : args.learning,
		'train_batch_sz' : args.train_batch_sz,
		'valid_batch_sz' : args.valid_batch_sz,
		'seed' : args.seed,
		'wandb' : bool(args.wandb)
	}

	print(f'Config Used : {config}')


	if config['action'] == 'train':

		if config['wandb']:
			wandb.init(project='twitter-sentiment', config=config)

		setup_train(config['epochs'], config['learning'], config['train_batch_sz'], config['valid_batch_sz'], config['seed'], is_wandb=config['wandb'])

	elif config['action'] == 'test':

		setup_test()