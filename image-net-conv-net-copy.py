import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from pysrc.zipper import *
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets
import pickle

PATH = "./data/weights/image-net4.pt"
device = torch.device("cpu") 

def fit(epochs, model, loss_func, opt, train_dl):
    print('Starting Training')

    for epoch in tqdm(range(epochs)):  
        running_loss = 0.0
        model.train()
        for input, label in tqdm(train_dl):
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = loss_func(output, label)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        print(f"Epoch {epoch + 1}/{epochs} Training Loss: {loss.item():.4f}")
    
    print('Finished Training')


def accuracy(model, test_dl):
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for data in tqdm(test_dl):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the validation set: %d %%' % (
        100 * correct / total))
    

# Model
class ConvQuantNet(nn.Module):
    def __init__(self):
        super(ConvQuantNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 25, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(25, 50, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(50, 70, 3, stride=2, padding=1)

        self.pool1 = nn.MaxPool2d(2, padding=1)
        self.pool2 = nn.MaxPool2d(2, padding=1)
        self.pool3 = nn.MaxPool2d(2, padding=1)

        self.bn1 = nn.BatchNorm2d(50)
        self.bn2 = nn.BatchNorm2d(70)

        self.fc1 = nn.Linear(4480, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 10)

        self.quant1 = torch.ao.quantization.QuantStub(qconfig=None)
        self.dequant1 = torch.ao.quantization.DeQuantStub(qconfig=None)
        self.quant2 = torch.ao.quantization.QuantStub(qconfig=None)
        self.dequant2 = torch.ao.quantization.DeQuantStub(qconfig=None)
        self.quant3 = torch.ao.quantization.QuantStub(qconfig=None)
        self.dequant3 = torch.ao.quantization.QuantStub(qconfig=None)
        self.quant4 = torch.ao.quantization.QuantStub(qconfig=None)
        self.dequant4 = torch.ao.quantization.DeQuantStub(qconfig=None)
        self.quant5 = torch.ao.quantization.QuantStub(qconfig=None)
        self.dequant5 = torch.ao.quantization.DeQuantStub(qconfig=None)
        self.quant6 = torch.ao.quantization.QuantStub(qconfig=None)
        self.dequant6 = torch.ao.quantization.DeQuantStub(qconfig=None)

    def forward(self, x):
        x = self.quant1(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.dequant1(x)

        x = self.quant2(x)
        x = self.pool2(self.bn1(F.relu(self.conv2(x))))
        x = self.dequant2(x)

        x = self.quant3(x)
        x = self.pool3(self.bn2(F.relu(self.conv3(x))))
        x = self.dequant3(x)
        x = x.reshape(x.size(0), -1)
        

        x = self.quant4(x)
        x = F.relu(self.fc1(x))
        x = self.dequant4(x)
        
        x = self.quant5(x)
        x = F.relu(self.fc2(x))
        x = self.dequant5(x)
        x = F.dropout(x, 0.25)
        x = self.quant6(x)
        x = F.softmax(self.fc3(x), dim=1)
        x = self.dequant6(x)
        return x

def training_loop(model, train_dl, epochs = 5, lr = 1e-4):
    loss_func = F.cross_entropy
    opt = optim.Adam(model.parameters(), lr=lr)
    # opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    fit(epochs, model, loss_func, opt, train_dl)
    
    
def main():
    # Define transformations

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
        
    print("Loading Data:")
    # Load the ImageNet dataset
    train_dataset = datasets.ImageFolder("imagenette2/train/", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    val_dataset = datasets.ImageFolder("imagenette2/val/", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    print(len(train_dataset))
    print(len(val_dataset))
    print(len(train_loader))
    print(len(val_loader))
    
    print("Done With Loading Data")
    torch.backends.quantized.engine = 'qnnpack'
    net = ConvQuantNet()
    
    model_fp32= ConvQuantNet()
   
    net.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32)
   
    training_loop(model_fp32_prepared, train_loader, epochs=1, lr=1e-4)
    
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    # print(model_int8)
    print("First Accuracy of model_int8:")
    accuracy(model_int8, val_loader)
    print("Second Accuracy of model_fp32_prepared:")
    accuracy(model_fp32_prepared, val_loader)
    print("Saving model")
    torch.save(model_int8.state_dict(), PATH)
    # pickle.dump(model_int8, open(PATH, 'wb'))

if __name__ ==  '__main__':
    main()