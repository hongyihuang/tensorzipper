import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from pysrc.zipper import *
from tqdm import tqdm

PATH = "./data/weights/cifar-fc.pt"
device = torch.device("cpu") #torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 4. Train the network
def fit(epochs, model, loss_func, opt, train_dl):
    print('Starting Training')
    model.train()
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for input, label in train_dl:
            input = input.to(device)
            label = label.to(device)
            opt.zero_grad()
            output = model(input)
            loss = loss_func(output, label)
            loss.backward()
            opt.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/train_dl.__len__()}")

    print('Finished Training')

def training_loop(model, train_dl, loss_func = F.cross_entropy, epochs = 20, lr = 0.001):
    opt = optim.Adam(model.parameters(), lr=lr)
    fit(epochs, model, loss_func, opt, train_dl)

# 5. Test the network
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

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

class FCQuantNet(nn.Module):
    def __init__(self):
        super(FCQuantNet, self).__init__()
        self.fc1 = nn.Linear(3 * 32 * 32, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 10)
        self.relu6 = torch.nn.ReLU6()
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.Softmax(1)
        
        self.quant = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant = torch.ao.quantization.DeQuantStub(qconfig = None)

        self.quant2 = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant2 = torch.ao.quantization.DeQuantStub(qconfig = None)

        self.quant3 = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant3 = torch.ao.quantization.DeQuantStub(qconfig = None)

    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        x = self.quant(x)
        x = self.relu6(self.fc1(x))
        x = self.dequant(x)
        
        x = self.quant2(x)
        x = self.relu6(self.fc2(x))
        x = self.dequant2(x)

        x = self.quant3(x)
        x = self.fc3(x)
        x = self.dequant3(x)
        x = self.softmax(x)
        return x

def main():
    # See turorial https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    # 1. Load the CIFAR-10 dataset
    transform = transforms.Compose([
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transformTest = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transformTest)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
    torch.backends.quantized.engine = 'qnnpack'
    net = FCQuantNet()
    net.to(device)

    net.eval()

    net.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    model_fp32_prepared = torch.ao.quantization.prepare_qat(net.train())

    training_loop(model_fp32_prepared, trainloader, epochs=10, lr=0.0001)
    training_loop(model_fp32_prepared, trainloader, epochs=10, lr=0.00001)

    model_fp32_prepared.eval()
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

    accuracy(model_int8, testloader)

    accuracy(model_fp32_prepared, testloader)
    
    torch.save(model_int8, PATH)

if __name__ ==  '__main__':
    main()