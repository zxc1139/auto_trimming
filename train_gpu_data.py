import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import glob
from pathlib import Path
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from keras.utils.data_utils import pad_sequences
import segmentation_models_pytorch as smp
from dataset_gpu import GPU_Dataset
from model import RNNModel, FCModel, TransformerModel
import argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_visualization(n_epochs, val_loss_list, train_loss_list):
    epoch = range(1, n_epochs+1)
    plt.grid(True, which ="both")
    plt.semilogy(epoch, val_loss_list, 'b', label='Validation Loss')
    plt.semilogy(epoch, train_loss_list, 'r', label='Training Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

def input_visualization(x, y):
    plt.plot(x, y)
    plt.xlabel("relative time (start_time = 0)")
    plt.ylabel("FPS")
    plt.title("Sample input data visualization")
    plt.show()

def main() -> None: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Transformer', type=str, help='type of network (FC, RNN, LSTM, GRU, Transformer)')
    args= parser.parse_args()

    gpu_data_path = Path(r".\reddeadredemption2\reddeadredemption2_gpu_data")
    trimming_path = Path(r".\reddeadredemption2\reddeadredemption2_trimming")
    whole_path = Path(r".\reddeadredemption2\reddeadredemption2_whole")
    session_dataset = GPU_Dataset(gpu_data_path, whole_path, trimming_path)
    input_0, target_0 = session_dataset[1]
    target_0 = torch.squeeze(target_0)

    
    input_size =  input_0.shape[1]
    layer_dim = 8 #num of encoder layer 
    output_size = target_0.shape[0]
    activation_function = 'relu'
    lr = 0.001
    num_epochs = 300
    b_size = 48
    nhead = 2 #4
    nfeed = 2048
    hidden_dim = 512
    dropout1 = 0.3
    dropout2 = 0.1

    validation_size = int(0.2 * len(session_dataset))
    training_size = len(session_dataset) - validation_size
    print(training_size, validation_size)
    print(input_size)

    train_dataset, validation_dataset = torch.utils.data.random_split(session_dataset, [training_size, validation_size])
    train_dataloader = DataLoader(train_dataset, batch_size=b_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=b_size, shuffle=True)

    one_batch = next(iter(train_dataloader))
    input_norm, target_norm = one_batch

    if args.model == 'RNN':
        model = RNNModel(input_dim = input_size, hidden_dim = hidden_dim, layer_dim = layer_dim, 
                        output_dim = output_size, activation_fun = activation_function).to(DEVICE)
    if args.model == 'FC':
        model = FCModel(input_dim = input_size, hidden_dim = hidden_dim, output_dim = output_size).to(DEVICE)
    if args.model == 'Transformer':
        model = TransformerModel(input_dim = input_size, nhead = nhead, nlayers = layer_dim, 
                                 nfeed = nfeed,output_dim = output_size, dropout1 = dropout1,
                                 dropout2 = dropout2, hidden_dim = hidden_dim).to(DEVICE)
    print(model)
    y = torch.squeeze(input_norm[0], 0)
    x = np.arange(start=0, stop=len(y), step=1)
    input_visualization(x, y) 

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,150,240], gamma=0.1)

    training_loss_min = np.Inf # track change in validation loss
    validation_loss_min = np.Inf # track change in validation loss
    train_loss_list = []
    val_loss_list = []
    
    # EXPERIMENT_NAME = "reddeadredemption2_transformer_run_0"
    # EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    # mlflow.set_tracking_uri('http://localhost:5000')
    # mlflow.set_experiment(EXPERIMENT_NAME)
    # with mlflow.start_run():
    import time 
    start_time = time.time()
    for epoch in range(1, num_epochs+1):
        # keep track of training and validation loss
        training_loss = 0.0
        validation_loss = 0.0
        # fc_net.train()
        model.train()
        for batch_idx, (inputs, targets) in enumerate (train_dataloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            # calculate the batch loss
            loss = criterion(outputs, targets)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            training_loss += loss.item()*inputs.size(0)
            
        # calculate average losses
        training_loss = training_loss/len(train_dataloader.sampler)
        train_loss_list.append(training_loss)

        model.eval()
        for batch_idx, (inputs, targets) in enumerate (validation_dataloader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)
            outputs_val = model(inputs)
            
            loss_val = criterion(outputs_val, targets)
            validation_loss += loss_val.item()*inputs.size(0)
            
        # calculate average losses
        validation_loss = validation_loss/len(validation_dataloader.sampler)
        val_loss_list.append(validation_loss)
        
        scheduler.step()
        
        print('Epoch: {}/{}  \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, num_epochs, training_loss, validation_loss))
        print("Current learning rate : {}".format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        model_name = "forza5_transformer"
        # save model if validation loss has decreased
        if validation_loss <= validation_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            validation_loss_min,
            validation_loss))        
            # torch.save(fc_net.state_dict(), model_name)
            torch.save(model.state_dict(), model_name)
            validation_loss_min = validation_loss
                
        time_delta = time.time() - start_time
        print("--- Training time: %s ---" % (time.strftime('%H:%M:%S', time.gmtime(time_delta))))
        print("Training done")

    loss_visualization(num_epochs, val_loss_list, train_loss_list)


if __name__ == '__main__':
    main()