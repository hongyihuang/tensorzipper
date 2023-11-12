import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from pysrc.zipper import *
from pysrc.torchQuant import Q_int8
from tqdm import tqdm

PATH = "./data/weights/cifar-fc.pt"
BITS = 16

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
        x = self.dropout(self.relu6(self.fc1(x)))
        x = self.dequant(x)
        
        x = self.quant2(x)
        x = self.dropout(self.relu6(self.fc2(x)))
        x = self.dequant2(x)

        x = self.quant3(x)
        x = self.fc3(x)
        x = self.dequant3(x)
        x = self.softmax(x)
        return x
    
def main():
    torch.backends.quantized.engine = 'qnnpack'

    model = torch.load(PATH)
    model.eval()

    transformTest = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transformTest)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=2)
    
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # get some testing images
    dataiter = iter(testloader)
    image, label = next(dataiter)

    # show images
    # imshow(torchvision.utils.make_grid(images[0:8]))
    # print labels
    print(classes[label[0]])

    outputs = model.forward(image[0])
    _, predicted = outputs.max(1)
    predicted = predicted.detach().numpy()
    print(outputs)
    print(classes[predicted[0]])

    # PYTHONIZED C ALGORITHM, CHECK AGAINST PYTORCH
    print("INPUT: ")
    print(model.quant(image.view(-1, 3*32*32)))
    print(model)
    img = np.round(image.detach().numpy()/0.0078).astype(np.int8).flatten()
    input = np.round(image.view(-1, 3*32*32).detach().numpy()/0.0078).astype(np.int8)
    print(input[0])
    print("Diff: ", np.sum(np.abs(input[0]-img)))
    
    print("fc1")
    layer = model.fc1
    fc1 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    fc1 = np.minimum(7, fc1)
    fc1 = np.maximum(-8, fc1)
    print(fc1.shape)
    print("Nonzeros:", np.count_nonzero(fc1))

    fcMul = (fc1 @ input.transpose()).transpose()

    print("h1")
    h1 = model.quant2(model.dequant(model.relu6(model.fc1(model.quant(image.view(-1, 3 * 32 * 32))))))
    print(h1[0])

    print("Ours:")
    print(fcMul.shape)
    print(model.quant2.scale)
    M1 = (model.quant.scale * model.fc1.weight().q_scale()*16 / model.quant2.scale).item()
    M1 = round(M1*(2**16))
    b_1 = np.round(model.fc1.bias().detach().numpy() / model.quant2.scale.item()).astype(np.int8)

    ourResults = ((((fcMul << 0)*M1)>>BITS) + b_1)
    ourResults = np.maximum(0, ourResults)
    ourResults = np.minimum(round(6 / model.quant2.scale.item()), ourResults)

    print(np.round(ourResults * model.quant2.scale.item(), 4)[0, :16])
    print(ourResults[0])

    diff = h1.dequantize().numpy() - (ourResults * model.quant2.scale.item())
    print("Avg diff: ", np.sum(np.abs(diff))/1000)

    print("h2")
    h2 = model.quant3(model.dequant2(model.relu6(model.fc2(h1))))
    print(h2.dequantize()[0])

    M2 = (model.quant2.scale * model.fc2.weight().q_scale()*16 / model.quant3.scale).item()
    M2 = round(M2*(2**16))
    b_2 = np.round(model.fc2.bias().detach().numpy() / model.quant3.scale.item()).astype(np.int8)
    
    layer = model.fc2
    fc2 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    ourResults = ((((fc2 @ ourResults.transpose()).transpose() << 0)*M2)>>BITS) + b_2
    ourResults = np.maximum(0, ourResults)
    ourResults = np.minimum(round(6 / model.quant3.scale.item()), ourResults)
    
    print(np.round(ourResults[0, :16] * (model.quant3.scale.item()), 4))
    print(ourResults[0])

    diff = h2.dequantize().numpy() - (ourResults * model.quant3.scale.item())
    print("Avg diff: ", np.sum(np.abs(diff))/1000)

    print("h3")
    h3 = model.fc3(h2)
    print(h3.dequantize()[0])

    M3 = (model.quant3.scale * model.fc3.weight().q_scale()*16).item()
    M3 = round(M3*(2**16))
    b_3 = np.round(model.fc3.bias().detach().numpy()).astype(np.int8)

    layer = model.fc3
    fc3 = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int32)
    ourResults = (((fc3 @ ourResults.transpose()).transpose()*M3)>>BITS) + b_3
    
    print(ourResults[0])

    diff = h3.dequantize().numpy() - (ourResults)
    print("Avg diff: ", np.sum(np.abs(diff))/1000)
    print("M1 = ", M1)
    print("M2 = ", M2)
    print("M3 = ", M3)

if __name__ ==  '__main__':
    main()