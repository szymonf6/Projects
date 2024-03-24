import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#getting a dataset
train_data = datasets.FashionMNIST(
    root="data", #where to download the data
    train=True, #do we want the training datasets
    download=True,
    transform=ToTensor(), #how do we want to transform the data
    target_transform=None #how do we want to transform thr labels/targets

)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

#how many samples do we have
#print(len(train_data), len(test_data))

#see the training example
class_names = train_data.classes
#print(class_names)


#visualize0
image, label = train_data[0]
#print(image.shape)
#plt.imshow(image.squeeze(), cmap="gray") 
#plt.show()

#torch.manual_seed(42)
#fig = plt.figure(figsize=(9,9))
#rows, cols = 4, 4
#for i in range (1, rows*cols+1):
    #random_idx = torch.randint(0, len(train_data), size=[1]).item()
    #img, label = train_data[random_idx]
    #fig.add_subplot(rows, cols, i)
    #plt.title(class_names[label])
    #plt.imshow(img.squeeze(), cmap="gray")


#plt.show()
    
#setup the batch size hyperparameter
BATCH_SIZE = 32

#turn datasets into iterables
train_dataLoader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataLoader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

#check out
#print(train_dataLoader)
#print(test_dataLoader)

#check whats inside the training dataloader
#pobieramy pierwszy batch danych
#iter zwraca iterator z podanego argumentu z DataLoadera, next jest używane do pobrania następnego elementu z tego iteratora
#i tutaj wzrocimy 1 element, jakbyśmy w kolejnej linijce dali znowu next(iter()) to zwrócimy drugi element itd.
train_features_batch, train_labels_batch = next(iter(train_dataLoader))

#printujemy rozmiar tensorów
#w train_features_batch mamy rozmiar [32, 1, 28, 28] - 32 - batch size, 1 - liczba kanałów(1 oznacza czarno-biały), 28x28 - rozmiar obrazu
# w train_labels_batch mamy [32] i odnosi się to do 32 etykiet 32 batcgy
#print(train_features_batch.shape)
#print(train_labels_batch.shape)

#show a sample
#torch.manual_seed(42)
#randint zwraca randowowego integera. randint(low, high, size, dtype=None, geenrator=None)
#low - najniższa możliwa wartość
#high - najwyższa możliwa wartość
#size - kształt tensora 
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
#print(f"Image shape: {img.shape}")
#print(f"Label: {label}, label size: {label.shape}")
#plt.show()


#build a baseline model
#create a flatten model
flatten_model = nn.Flatten()

x = train_features_batch[0]

#flatten the sample
output = flatten_model(x) #perform forward pass

class FashionMNISTModelV0(nn.Module):
    def __init__(self,
                 input_shape: int,
                 hidden_units: int,
                 output_shape: int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,
                      out_features=hidden_units),
            nn.Linear(in_features=hidden_units,
                      out_features=output_shape)
        )

    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)
model_0 = FashionMNISTModelV0(
    input_shape = 784, 
    hidden_units  = 10,
    output_shape=len(class_names)
)

#print(model_0)

dummy_x = torch.rand([1, 1, 28, 28])
print(model_0(dummy_x))

#setup loss optimizer adn evaluation metrics
import requests
from pathlib import Path

if Path("helper_functions.py").is_file():
    print("helper functions już są")
else:
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)

from helper_functions import accuracy_fn

#setup loss fn and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.1)

#create a function to time out our experiment
from timeit import default_timer as timer
def print_train_time(start: float,
                     end: float):
    #"print difference between start and end."
    total_time = end - start
    print(f"Time: {total_time:.3f} seconds")
    return total_time

#creating a training loop and training a model on batches of data
#1. loop through epochs
#2. loop through training batches, perform trining steps, calculate the train loss
#3. loop through testing batches, perform testing steps, calculate the lopp
#4. print out what's happening
#5. time it all

#import tqdm for progress bar
from tqdm.auto import tqdm

#set the seed and start the timer
torch.manual_seed(42)
train_time_start = timer()

#set number of epochs
epochs = 3

#create training and test loop
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch} \n-----")
    #training
    train_loss = 0
    #add a loop to loop through the training batches
    for batch, (X, y) in enumerate(train_dataLoader):
        model_0.train()
        #forward pass
        y_pred = model_0(X)

        #calculate the loss
        loss = loss_fn(y_pred, y)
        train_loss += loss #accumulate the training loss every batch

        #optimizer zero grad
        optimizer.zero_grad()

        #loss bacjward
        loss.backward()

        #optimizer step
        optimizer.step()

        #print out whats happening
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/{len(train_dataLoader.dataset)} samples.")
    
    #divide total train loss by length of train dataloader
    train_loss /= len(train_dataLoader)


    #testing loop
    test_loss, test_acc = 0, 0
    model_0.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataLoader:
            #1forward pass
            test_pred = model_0(X_test)

            #calcuate loss
            test_loss += loss_fn(test_pred, y_test)

            #calculate the accuracy
            test_acc += accuracy_fn(y_true=y_test, y_pred = test_pred.argmax(dim=1))
        
        #calculate the test loss average per batch
        test_loss /= len(test_data)

        #calculate the test acc average per batch
        test_acc /= len(test_dataLoader)

    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")


#calculate training time
train_time_end = timer()
total_train_time_model_0 = print_train_time(start=train_time_start,
                                            end=train_time_end)


#make predictions and get model 0 results
torch.manual_seed(42)
def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn):
    #returna a dictionary containing the results of model predicting on data loader
    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            #make predictions
            y_pred = model(X)

            #accumulate the loss and acc values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y,
                               y_pred = y_pred.argmax(dim=1))
            
        #scale loss and acc to find average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model_name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_acc": acc}


#calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataLoader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)

print(model_0_results)