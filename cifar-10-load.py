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

    def quantizeInt16(n):
        return np.round(n*(2**16)).astype(np.int16)
    
    M1 = (model.quant.scale * model.fc1.weight().q_scale()*16 / model.quant2.scale).item()
    M2 = (model.quant2.scale * model.fc2.weight().q_scale()*16 / model.quant3.scale).item()
    M3 = (model.quant3.scale * model.fc3.weight().q_scale()*16).item()

    b_1 = np.round(model.fc1.bias().detach().numpy() / model.quant2.scale.item()).astype(np.int8)
    b_2 = np.round(model.fc2.bias().detach().numpy() / model.quant3.scale.item()).astype(np.int8)
    b_3 = np.round(model.fc3.bias().detach().numpy()).astype(np.int8)

    print("M values: ", round(M1*(2**16)), round(M2*(2**16)), round(M3*(2**16)))
    #print("b: ", b_1, b_2, b_3)
        
    print("int_repr")
    w = model.fc1.weight().int_repr().transpose(0, 1).detach().numpy()
    w = w>>4+(w>>3 & 1)
    #print("W: ", w.size, "\n", w)
    w = w.flatten()
    print("Compress: ")
    hist, bit = compress_nparr(w, 16)
    #print(hist)
    #print(len(bit))
    fc1_str = exportCArray(bit)
    #print(len(w))
    # +42% increase in performance and 42% decrease in storage
    # Before: (3072000/2/1024*3827)/240000000 ~0.024s
    # After: (893038/1024*3827)/240000000 ~0.014s
    print("% change: ", (893038-(3072000/2))/(3072000/2))
    print("% performance: ", (0.014-0.024)/0.024)


    # 3*32*32*128
    # print(fc1_str)

    print("Bias: ")
    bins = 16
    s_b = 2*torch.max(torch.abs(model.fc1.bias()))/bins
    b = Q_int8(model.fc1.bias(), s_b, 0).detach().numpy().astype(np.int8)
    print(b.size)
    #print(model.fc1.bias())

    # generate a .h file that contains: fc1, fc2, fc3 weights, size, and bias
    f = open("./runtime/esp32-s3/cifar-10-fc/cifar-10-fc.h", "w")

    Mstr = "const static uint16_t {} = {};\n"
    f.write(Mstr.format("M1", quantizeInt16(M1)))
    f.write(Mstr.format("M2", quantizeInt16(M2)))
    f.write(Mstr.format("M3", quantizeInt16(M3)))

    Mstr = "const static uint16_t {} = {};\n"
    f.write(Mstr.format("S1", round(1/model.quant.scale[0].item())))
    f.write(Mstr.format("S2", round(1/model.quant2.scale[0].item())))
    f.write(Mstr.format("S3", round(1/model.quant3.scale[0].item())))

    arrayStr = "const static int8_t {}[{}] = {};\n"
    f.write(arrayStr.format("fc1_b", len(b_1), exportCArray(b_1)))
    f.write(arrayStr.format("fc2_b", len(b_2), exportCArray(b_2)))
    f.write(arrayStr.format("fc3_b", len(b_3), exportCArray(b_3)))

    def emitCompress(template, name, layer):
        w = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int8)
        w = w.flatten()
        print(name)
        w = np.minimum(7, w)
        w = np.maximum(-8, w)
        print(np.min(w), np.max(w))
        print(w)
        w += 8

        hist, bit = compress_nparr(w, 16)
        
        return template.format(name+"_d", len(bit), exportCArray(bit)) + template.format(name+"_f", len(hist), exportCArray(hist))
    
    def emitInt4(template, name, layer):
        w = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int8)        
        w = w.flatten()
        print(name)
        w = np.minimum(7, w)
        w = np.maximum(-8, w)
        print(np.min(w), np.max(w))
        print(w)
        w += 8
        
        arr = np.zeros(len(w)>>1, dtype=np.uint8)
        for i in range(len(arr)):
            arr[i] = (w[i*2] & 15) + (w[(i*2)+1] << 4)
            assert w[i*2] == (arr[i].astype(np.int16) & 15)
            assert w[(i*2)+1] == (arr[i].astype(np.int16) >> 4)
        return template.format(name, len(arr), exportCArray(arr))
    
    arrayStr = "const uint8_t {}[{}] = {};\n"
    f.write(emitCompress(arrayStr, "fc1_w", model.fc1))
    f.write(emitCompress(arrayStr, "fc2_w", model.fc2))
    f.write(emitCompress(arrayStr, "fc3_w", model.fc3))

    arrayStr = "const uint8_t {}[{}] = {};\n"
    f.write(emitInt4(arrayStr, "fc1_w", model.fc1))
    f.write(emitInt4(arrayStr, "fc2_w", model.fc2))
    f.write(emitInt4(arrayStr, "fc3_w", model.fc3))

    transformTest = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transformTest)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                            shuffle=False, num_workers=1)
    
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # get some testing images
    dataiter = iter(testloader)
    image, label = next(dataiter)
    
    arrayStr = "const static int8_t {}[{}] = {};\n"
    img = np.round(image.detach().numpy()/0.0078).astype(np.int8).flatten()
    f.write(arrayStr.format("img", len(img), exportCArray(img)))

if __name__ ==  '__main__':
    main()