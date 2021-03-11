import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import re
import string
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import preprocessor as tweet_preprocess
from sklearn.metrics import accuracy_score, f1_score

from src.apost import APOSTOPHES

def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)



def replace_apostophes(text):

	apostophes = APOSTOPHES

	for keys in apostophes.keys():

		text = text.replace(keys, apostophes[keys])

	return text



def preprocess_old(text):
	try:
		text = re.sub(r"[^\x00-\x7F]"," ", text)
		text = re.sub(r"http\S+", " ", text) # Remove https urls
		text = replace_apostophes(text) # Replace apostophes
		text = re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ", text) # Remove hashtags and mentions
		text = remove_emoji(text) # Remove emojis
		text = re.sub(f"[{re.escape(string.punctuation)}0-9\\r\\t\\n]", " ", text) # Remove string punctuations, numbers and other escape characters
		text = text.lower()
		text = text.split(" ")
		text = list(filter(lambda x: x not in ['',' '], text))

		return " ".join(text)

	except Exception:
		print(text)


def preprocess(text, stem=True, lemmatize=False):



	text = tweet_preprocess.clean(text)
	text = text.lower()
	text = replace_apostophes(text)
	text = re.sub(f"[{re.escape(string.punctuation)}0-9\\r\\t\\n]", " ", text) # Remove string punctuations, numbers and other escape characters
	text = text.split(" ")
	text = list(filter(lambda x: x not in ['',' '], text))

	if lemmatize:
		text = [WordNetLemmatizer().lemmatize(w) for w in text]
		text = [WordNetLemmatizer().lemmatize(w, pos='v') for w in text]

	if stem:
		text = [SnowballStemmer('english').stem(w) for w in text]

	text = " ".join(text)		

	return text

def calculate_metrics(y_pred, y_test):

	y_pred = (y_pred > 0)

	y_pred = y_pred.detach().cpu().numpy()
	y_test = y_test.detach().cpu().numpy()

	acc = accuracy_score(y_test, y_pred)

	f1 = f1_score(y_test, y_pred)


	return acc, f1


def load_chkpt(model, device=torch.device("cpu"), file_path=None):

	if file_path == None:
		return


	print(f"Loading model from ==> {file_path}")

	state_dict = torch.load(file_path, map_location=device)

	print(f'Validation Loss : {state_dict["valid_loss"]}')
	print(f'Validation Acc : {state_dict["valid_acc"]}')
	print(f'Validation F1-Score : {state_dict["valid_f1"]}')

	model.load_state_dict(state_dict['model'])

	return model


def pad_sequence(text, fix_length=120):
	final_text = []

	if len(text) < fix_length:

		for i in range(fix_length - len(text)):

			final_text.append("<pad>")

		final_text.extend(text)

	elif len(text) > fix_length:
		final_text = text[:fix_length]

	else:
		final_text = text

	return final_text


def load_glove_vectors(file_path):

	word_vectors = {}

	with open(file_path, 'r') as f:

		for data in f:
			data = data.split()
			word_vectors[data[0]] = np.asarray([float(x) for x in data[1:]])


	return word_vectors


def get_glove_vectors(glove_vectors, vocab, embedd_size=50):

	vocab_size = len(vocab)

	W = np.zeros((vocab_size, embedd_size), dtype='float32')

	W[0] = np.random.uniform(-0.25, 0.25, embedd_size) # vector for <unk> token
	W[1] = np.zeros(embedd_size, dtype='float32') # vector for <pad> token


	i = 2

	for words in vocab:


		if words not in ["<unk>", "<pad>"]:

			W[i] = glove_vectors.get(words, W[1])
			i+= 1

	return torch.from_numpy(W)


def encode_sentence(vocab, text):

	encode = []

	for w in text:
		encode.append(vocab.get(w, vocab["<unk>"]))

	return torch.from_numpy(np.asarray(encode))