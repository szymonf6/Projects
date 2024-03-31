import torch
from torch import nn

import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from timeit import default_timer as timer
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

#fashion mnist model
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

#create a function to time out our experiment
def print_train_time(start: float,
                     end: float):
    #"print difference between start and end."
    total_time = end - start
    print(f"Time: {total_time:.3f} seconds")
    return total_time

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

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

#see the training example
class_names = train_data.classes

#visualize 
image, label = train_data[0]
print(image.shape)
plt.imshow(image.squeeze(), cmap="gray") 
plt.show()

torch.manual_seed(42)
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range (1, rows*cols+1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.title(class_names[label])
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
    
#setup the batch size hyperparameter
BATCH_SIZE = 32

#turn datasets into iterables
train_dataLoader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

test_dataLoader = DataLoader(dataset=test_data,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

#check whats inside the training dataloader
#pobieramy pierwszy batch danych
#iter zwraca iterator z podanego argumentu z DataLoadera, next jest używane do pobrania następnego elementu z tego iteratora
#i tutaj wzrocimy 1 element, jakbyśmy w kolejnej linijce dali znowu next(iter()) to zwrócimy drugi element itd.
train_features_batch, train_labels_batch = next(iter(train_dataLoader))

#show a sample
#randint zwraca randowowego integera. randint(low, high, size, dtype=None, geenrator=None)
#low - najniższa możliwa wartość
#high - najwyższa możliwa wartość
#size - kształt tensora 
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis(False)
print(f"Image shape: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
plt.show()

#build a baseline model
#create a flatten model
flatten_model = nn.Flatten()

x = train_features_batch[0]

#flatten the sample
output = flatten_model(x) #perform forward pass

torch.manual_seed(42)
model_0 = FashionMNISTModelV0(
    input_shape = 784, 
    hidden_units  = 10,
    output_shape=len(class_names)
)

dummy_x = torch.rand([1, 1, 28, 28])
print(model_0(dummy_x))

#setup loss fn and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.1)

#creating a training loop and training a model on batches of data
#1. loop through epochs
#2. loop through training batches, perform trining steps, calculate the train loss
#3. loop through testing batches, perform testing steps, calculate the lopp
#4. print out what's happening
#5. time it all

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
#calculate model 0 results on test dataset
model_0_results = eval_model(model=model_0,
                             data_loader=test_dataLoader,
                             loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)

print(model_0_results)