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
from torchvision import transforms
import pickle

PATH = "./data/weights/image-net2.pt"
device = torch.device("cpu") 


# Train the network
def fit(epochs, model, loss_func, opt, train_dl):
    print('Starting Training')
    model.train()
    for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
        running_loss = 0.0
        
        for input, label in train_dl:
            input = input.to(device)
            label = label.to(device)
            output = model(input)
            loss = loss_func(output, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            
        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs} Loss: {loss.item():.4f}")
    
    print('Finished Training')
    
def training_loop(model, train_dl, loss_func = F.cross_entropy, epochs = 20, lr = 0.0001):
    # opt = optim.Adam(model.parameters(), lr=lr)
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
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

    print('Accuracy of the network on the 1000 test images: %d %%' % (
        100 * correct / total))
    
class ConvQuantNet(nn.Module):
    def __init__(self):
        super(ConvQuantNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 32 * 32, 512)
        self.fc2 = nn.Linear(512, 200)  # ImageNet has 200 classes
        
        self.quant = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant = torch.ao.quantization.DeQuantStub(qconfig = None)

        self.quant2 = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant2 = torch.ao.quantization.DeQuantStub(qconfig = None)

        self.quant3 = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant3 = torch.ao.quantization.DeQuantStub(qconfig = None)
        self.quant4 = torch.ao.quantization.QuantStub(qconfig = None)
        self.dequant4 = torch.ao.quantization.DeQuantStub(qconfig = None)

    def forward(self, x):
       
        x = self.quant(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dequant(x)
        
        x = self.quant2(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dequant2(x)
        x = x.reshape(-1, 128 * 32 * 32)
        x = self.quant3(x)
        x = F.relu(self.fc1(x))
        x = self.dequant3(x)
        
        x = self.quant4(x)
        x = self.fc2(x)
        x = self.dequant4(x)
        return x
    
def main():
    # Define transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    print("Loading Data:")
    # Load the ImageNet dataset
    train_imagenet = load_dataset('zh-plus/tiny-imagenet', split='train')
    test_imagenet = load_dataset('zh-plus/tiny-imagenet', split='valid')
    print(len(train_imagenet))
    print(len(test_imagenet))
    train_images, train_labels = [], []
    test_images, test_labels = [], []
    
    for image, label in zip(train_imagenet[:1000]['image'], train_imagenet[:100000]['label']):
        # print(image)
        # print(label)
        train_images.append(transform(image))
        train_labels.append(label)
        
    for image, label in zip(test_imagenet[:1000]['image'], test_imagenet[:10000]['label']):
        test_images.append(transform(image))
        test_labels.append(label)
    # for entry in train_imagenet:
    #     train_images.append(transform(entry['image']))
    #     train_labels.append(entry['label'])
    # for entry in test_imagenet:
    #     test_images.append(transform(entry['image']))
    #     test_labels.append(entry['label'])
        
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels)
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)

    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle= False)
    
    print(len(train_images))
    print(len(train_labels))
    print(len(test_images))
    print(len(test_labels))
    
    print("Done With Loading Data")
    
    
    torch.backends.quantized.engine = 'qnnpack'
    net = ConvQuantNet()
    net.to(device)

    net.eval()

    net.qconfig = torch.ao.quantization.get_default_qat_qconfig('qnnpack')
    torch.backends.quantized.engine = 'qnnpack'

    model_fp32_prepared = torch.ao.quantization.prepare_qat(net.train())
    # print("Training with lr=0.0001")
    # training_loop(model_fp32_prepared, train_loader, epochs=5, lr=0.0001)
    # print("Training with lr=0.00001")
    # training_loop(model_fp32_prepared, train_loader, epochs=5, lr=0.00001)

    # print("Running fit separately with lr=0.0005 and momentum=0.9:")
    # fit(1, model_fp32_prepared, F.cross_entropy, optim.SGD(model_fp32_prepared.parameters(), lr=0.0005, momentum=0.9), train_loader)

    model_fp32_prepared.eval()
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    # print(model_int8)
    print("First Accuracy:")
    accuracy(model_int8, test_loader)
    print("Second Accuracy:")
    accuracy(model_fp32_prepared, test_loader)
    print("Saving model")
    # torch.save(model_int8, PATH)
    pickle.dump(model_int8, open(PATH, 'wb'))

if __name__ ==  '__main__':
    main()