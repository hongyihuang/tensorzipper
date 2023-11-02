import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from pysrc.zipper import *
from pysrc.torchQuant import Q_int8
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset


PATH = "./data/weights/image-net.pt"

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
    torch.backends.quantized.engine = 'qnnpack'

    model = torch.load(PATH)
    model.eval()
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    
    print("Loading TestData:")
    # Load the ImageNet dataset
    test_imagenet = load_dataset('zh-plus/tiny-imagenet', split='valid')
    print(len(test_imagenet))
  
    test_images, test_labels = [], []
    
    for image, label in zip(test_imagenet[:1000]['image'], test_imagenet[:1000]['label']):
        test_images.append(transform(image))
        test_labels.append(label)
        
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels)

    test_dataset = TensorDataset(test_images, test_labels)

    test_loader = DataLoader(test_dataset, batch_size=128)
    
    print(len(test_images))
    print(len(test_labels))
    
    print("Done With Loading Test Data")
    
    # get some testing images
    dataiter = iter(test_loader)
    image, label = next(dataiter)
    
    # show images
    # imshow(torchvision.utils.make_grid(images[0:8]))
    # print labels
    # print(classes[label[0]])

    outputs = model.forward(image[0])
    _, predicted = outputs.max(1)
    predicted = predicted.detach().numpy()
    # print(outputs)
    # print(classes[predicted[0]])
    
    
    # TODO PYTHONIZE THE C ALGORITHM, CHECK AGAINST PYTORCH
    # print("INPUT: ")
    # print(model.quant(image.view(-1, 3*32*32)))
    # print(model)
    img = np.round(image.detach().numpy()/0.0078).astype(np.int8).flatten()
    input = np.round(image.view(-1, 3*32*32).detach().numpy()/0.0078).astype(np.int8)
    # print(input[0])
    # print("Diff: ", np.sum(np.abs(input[0]-img)))
    
    # print("fc1")
    layer = model.conv1
    conv1 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    conv1 = np.minimum(7, conv1)
    conv1 = np.maximum(-8, conv1)
    # print(fc1.shape)

    conv1Mul = (conv1 @ input.transpose()).transpose()

    print("h1")
    h1 = model.quant2(model.dequant(model.relu6(model.conv1(model.quant(image.view(-1, 3 * 32 * 32))))))
    print(h1[0])

    # print("Ours:")
    # print(fcMul.shape)
    # print(model.quant2.scale)
    M1 = (model.quant.scale * model.conv1.weight().q_scale()*16 / model.quant2.scale).item()
    M1 = round(M1*(2**16))
    b_1 = np.round(model.conv1.bias().detach().numpy() / model.quant2.scale.item()).astype(np.int8)

    ourResults = (((conv1Mul*M1)>>16) + b_1)
    ourResults = np.maximum(0, ourResults)
    ourResults = np.minimum(round(6 / model.quant2.scale.item()), ourResults)

    # print(np.round(ourResults * model.quant2.scale.item(), 4)[0, :16])
    # print(ourResults[0])

    diff = h1.dequantize().numpy() - (ourResults * model.quant2.scale.item())
    print("Avg diff: ", np.sum(np.abs(diff))/1000)

    # print("h2")
    h2 = model.quant3(model.dequant2(model.relu6(model.conv2(h1))))
    # print(h2.dequantize()[0])

    M2 = (model.quant2.scale * model.fc2.weight().q_scale()*16 / model.quant3.scale).item()
    M2 = round(M2*(2**16))
    b_2 = np.round(model.conv2.bias().detach().numpy() / model.quant3.scale.item()).astype(np.int8)
    
    layer = model.conv2
    conv2 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    ourResults = (((conv2 @ ourResults.transpose()).transpose()*M2)>>16) + b_2
    ourResults = np.maximum(0, ourResults)
    ourResults = np.minimum(round(6 / model.quant3.scale.item()), ourResults)
    
    # print(np.round(ourResults[0, :16] * (model.quant3.scale.item()), 4))
    # print(ourResults[0])

    diff = h2.dequantize().numpy() - (ourResults * model.quant3.scale.item())
    print("Avg diff: ", np.sum(np.abs(diff))/1000)

    # print("h3")
    h3 = model.fc1(h2)
    # print(h3.dequantize()[0])

    M3 = (model.quant3.scale * model.fc1.weight().q_scale()*16).item()
    M3 = round(M3*(2**16))
    b_3 = np.round(model.fc1.bias().detach().numpy()).astype(np.int8)

    layer = model.fc1
    fc1 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    ourResults = (((fc1 @ ourResults.transpose()).transpose()*M3)>>16) + b_3
    
    # print(ourResults[0])

    diff = h3.dequantize().numpy() - (ourResults)
    print("Avg diff: ", np.sum(np.abs(diff))/1000)
    
    # print("h4")
    h4 = model.fc2(h3)
    # print(h3.dequantize()[0])

    M4 = (model.quant4.scale * model.fc2.weight().q_scale()*16).item()
    M4 = round(M4*(2**16))
    b_4 = np.round(model.fc2.bias().detach().numpy()).astype(np.int8)

    layer = model.fc2
    fc2 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    ourResults = (((fc2 @ ourResults.transpose()).transpose()*M4)>>16) + b_4
    
    # print(ourResults[0])

    diff = h4.dequantize().numpy() - (ourResults)
    print("Avg diff: ", np.sum(np.abs(diff))/1000)

if __name__ ==  '__main__':
    main()