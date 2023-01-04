import argparse
import sys
import torch.nn.functional as F
import torch
import click

import data
from torch.utils.data import DataLoader

from model import MyAwesomeModel
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

#from data import mnist



input_size = 784
hidden_layers = [512, 384, 256, 128, 64, 32]
output_size = 10
drop_p = 0.2


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = MyAwesomeModel(checkpoint['input_size'],
                             checkpoint['output_size'],
                             checkpoint['hidden_layers'], 
                             checkpoint['drop_p'])
    model.load_state_dict(checkpoint['state_dict'])
    return model
    

@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')



def train(lr):
    print("Training day and night")
    print(lr)

 


    # TODO: Implement training loop here
    model = MyAwesomeModel(input_size, output_size, hidden_layers, drop_p)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_dataloader = DataLoader(data.mnist("train", transform = data.ToTensor_and_norm()), batch_size = 16, shuffle = False)
    epochs = 10
    epochs_list = np.arange(0,epochs)
    train_losses = []
    for e in range(epochs):
        running_loss = 0
        for sample in train_dataloader:
            
            images = sample['image']
            labels = sample['labels']
            optimizer.zero_grad()
        
            output_logits = model(images)
 #           print(output_logits)
            loss = criterion(output_logits, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        print('Training loss is', running_loss)
        train_losses.append(running_loss)
        print(type(running_loss))
    
    checkpoint = {'input_size': 784,
              'output_size': 10,
              'hidden_layers': [512, 384, 256, 128, 64, 32],
              'drop_p': 0.2,
              'state_dict': model.state_dict()}

    torch.save(checkpoint, 'checkpoint_Mnist_model.pth')
    print(train_losses)   

    plt.plot(epochs_list, np.array(train_losses))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training loss as a function of epochs')
    plt.show()


    return train_losses



@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = load_checkpoint(model_checkpoint)
    model.eval()
    test_dataloader = DataLoader(data.mnist("test", transform = data.ToTensor_and_norm()), batch_size = 16, shuffle = False)
    accuracies = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        
        running_loss = 0
        for sample in test_dataloader:
            images = sample['image']
            labels = sample['labels']
            output_logits = model(images)

            loss = criterion(output_logits, labels)
            ps_val = F.softmax(output_logits, dim = 1)

            top_p_val, top_class_val = ps_val.topk(1,dim=1)

            equals = top_class_val == labels.view(top_class_val.shape)

            accuracy = torch.mean(equals.type(torch.FloatTensor))

            accuracies.append(accuracy)

            running_loss += loss.item()

        print('Test loss is', running_loss)
        print(f'Accuracy: {np.mean(accuracies)*100}%')
cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    