import numpy as np
import torch 
import torch.nn as nn
import os
import sys
import time
from tqdm import tqdm
import wandb


from utils import calculate_metrics



def save_chkpt(model, optimizer, valid_loss, valid_acc, valid_f1, file_path=None):

	if file_path == None:
		return

	if not os.path.exists(file_path):
		os.mkdir(file_path)

	state_dict = {
	'model' : model.state_dict(),
	'optimizer' : optimizer.state_dict(),
	'valid_loss' : valid_loss,
	'valid_acc' : valid_acc,
	'valid_f1' : valid_f1
	}

	torch.save(state_dict, os.path.join(file_path, 'model.pt'))

	print(f'Model saved to ==> {file_path}')



def train(model, optimizer, train_iter, valid_iter, criterion=nn.BCEWithLogitsLoss(), epochs=20, 
	best_valid_loss = np.float("Inf"), device=torch.device("cpu"), file_path=None, islog=None):
	
	torch.cuda.empty_cache()

	print(f'Device in use : {device}')
	
	train_losses = []
	valid_losses = []

	valid_acc = []
	valid_f1 = []


	for epoch in range(epochs):

		tloss = []
		vloss = []
		vacc = []
		vf1 = []

		model.train()

		for data in tqdm(train_iter):

			text, text_len = data.text
			targets = data.target

			targets = targets.reshape(-1, 1)

			text = text.to(device)
			targets = targets.to(device)

			optimizer.zero_grad()

			pred = model(text, text_len)

			loss = criterion(pred, targets)

			loss.backward()
			optimizer.step()

			tloss.append(loss.item())


		model.eval()

		
		for data in tqdm(valid_iter):

			with torch.no_grad():
				text, text_len = data.text
				targets = data.target

				targets = targets.reshape(-1, 1)

				text = text.to(device)
				targets = targets.to(device)

				pred = model(text, text_len)

				loss = criterion(pred, targets)

				vloss.append(loss.item())

				acc, f1 = calculate_metrics(pred, targets)

				vacc.append(acc)
				vf1.append(f1)

		tloss = np.mean(tloss)
		vloss = np.mean(vloss)
		vacc = np.mean(vacc)
		vf1 = np.mean(vf1)

		print(f'Epoch : {epoch+1}/ {epochs}| Train loss : {tloss:.7f}| Valid Loss : {vloss:.7f}, Valid Acc : {vacc:.4f}, Valid F1-Score : {vf1:.4f}')

		train_losses.append(tloss)
		valid_losses.append(vloss)
		valid_acc.append(vacc)
		valid_f1.append(vf1)


		if best_valid_loss > vloss:

			best_valid_loss = vloss
			save_chkpt(model, optimizer, vloss, vacc, vf1, file_path=file_path)


		if islog:
			wandb.log({
				'val_loss' : vloss,
				'val_acc' : vacc,
				'val_f1' : vf1
				})


	return train_losses, valid_losses, valid_acc, valid_f1





