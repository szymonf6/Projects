import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from torch.autograd import Variable

# Przygotowanie transformacji danych
transform = transforms.Compose([
    transforms.ToTensor(),  # Konwersja obrazu na tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalizacja wartości pikseli
])

# Załadowanie zbioru danych MNIST
mnist_train = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

print(mnist_train)
print(mnist_test)

# Przygotowanie danych
train_loader = DataLoader(mnist_train, batch_size=64, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=64, shuffle=False)

# Wyświetlenie przykładowych obrazów
figure = plt.figure(figsize=(10,8))
cols, rows = 5, 5
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(mnist_train), size=(1,)).item()
    img, label = mnist_train[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(label)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
#plt.show()

#loaders
loaders = {
    'train' : torch.utils.data.DataLoader(mnist_train,
                                          batch_size = 100,
                                          shuffle = True),

    'test' : torch.utils.data.DataLoader(mnist_test,
                                         batch_size = 100,
                                         shuffle = True)
}

#model
class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output, x    # return x for visualization
    
model = model()
print(model)

# Definicja różnych funkcji straty i optymalizatorów
loss_functions = [nn.CrossEntropyLoss(), 
                  nn.NLLLoss(), 
                  nn.MSELoss()]

optimizers = [optim.SGD(model.parameters(), lr=0.1),
              optim.Adam(model.parameters(), lr=0.1),
              optim.RMSprop(model.parameters(), lr=0.1)]


#train the model
epochs = 5

def train(epochs, model, loaders):
    for loss_function in loss_functions:
        for optimizer in optimizers:

            loss_fn = loss_function
            model_optimizer = optimizer

            model.train()

            total_step = len(loaders['train'])

            for epoch in range(epochs):
                for i, (images, labels) in enumerate(loaders['train']):

                    b_x = Variable(images)  #batch x
                    b_y = Variable(labels)  #batch y

                    output = model(b_x)[0]
                    loss = loss_fn(output, b_y)
                    
                    # clear gradients for this training step   
                    model_optimizer.zero_grad()           
                    
                    # backpropagation
                    loss.backward()    
                    # apply gradients             
                    model_optimizer.step()                
                    
                    if (i+1) % 100 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch + 1, epochs, i + 1, total_step, loss.item()))
                        pass
                pass
            pass

train(epochs, model, loaders)

def test():
    model.eval()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in loaders['test']:
            test_output, last_layer = model(images)
            y_pred = torch.max(test_output, 1)[1].data.squeeze()
            accuracy = (y_pred == labels).sum().item() / float(labels.size(0))
            pass

test()

sample = next(iter(loaders['test']))
imgs, lbls = sample

actual_number = lbls[:10].numpy()
print(actual_number)

test_output, last_layer = model(imgs[:10])
y_pred = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(f'Prediction number: {y_pred}')
print(f'Actual number: {actual_number}')


#1. create models directory
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

#2.create a model save path
MODEL_NAME = "workflow_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#3. SAVE THE MODEL STATE DICT
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model.state_dict(),
           f=MODEL_SAVE_PATH)


#loading a model
print(model.state_dict())
loaded_model = model()
loaded_model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

print(loaded_model.state_dict())
