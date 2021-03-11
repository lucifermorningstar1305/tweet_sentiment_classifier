import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os
import sys
import time


from src.model import Classifier
from src.utils import preprocess, load_glove_vectors, get_glove_vectors, pad_sequence, load_chkpt, encode_sentence

def predict(text):

	text = preprocess(text)
	text = text.split()

	text = pad_sequence(text)

	vocab_df = pd.read_csv("./data/vocab.csv")
	vocab = {k:v for k, v in zip(vocab_df['words'].values, vocab_df['index'].values)}

	encoded_sent = encode_sentence(vocab, text)

	embedd_mat = None

	if not os.path.exists("./MODELS/glove100d.pt"):

		glove_vec = load_glove_vectors("./MODELS/glove.6B.100d.txt")
		embedd_mat = get_glove_vectors(glove_vec, vocab, embedd_size=100)
		torch.save(embedd_mat, "./MODELS/glove100d.pt")

	else:
		embedd_mat = torch.load("./MODELS/glove100d.pt", map_location=torch.device("cpu"))

	device = torch.device("cpu")
	model = Classifier(len(vocab), 100, hidden_dim=128, embedd_vec=embedd_mat, device=device, is_trainable=False)
	model = load_chkpt(model, file_path="./MODELS/model.pt").to(device)
	model.eval()
	print("Model Load complete...")

	encoded_sent = encoded_sent.unsqueeze(0)

	y_pred = model(encoded_sent, len(encoded_sent))
	print(y_pred)
	y_pred = (y_pred > 0)
	y_pred = y_pred.detach().numpy()
	y_pred = int(y_pred.reshape(-1)[0])

	prediction = {0:"Negative", 1:"Positive"}
	
	return prediction[y_pred]

