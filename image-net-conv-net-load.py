import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
from pysrc.zipper import *
from pysrc.torchQuant import Q_int8
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import torch.nn.functional as F
import pickle
from keras.models import load_model

PATH = "./data/weights/image-net4.pt"

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
    
def main():
     
    torch.backends.quantized.engine = 'qnnpack'
    # model = load_model("model.keras")
    # model = torch.load(PATH)
    # model = pickle.load(open(PATH, 'rb'))
    # print(model)
    model = ConvQuantNet()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    
    
    def quantizeInt16(n):
        return np.round(n*(2**16)).astype(np.int16)
    
    
    print(model.quant1)
    M1 = (model.quant1.scale * model.conv1.weight().q_scale()*16 / model.quant2.scale).item()
    M2 = (model.quant2.scale * model.conv2.weight().q_scale()*16 / model.quant3.scale).item()
    M3 = (model.quant3.scale * model.fc1.weight().q_scale()*16 / model.quant4.scale).item()
    M4 = (model.quant4.scale * model.fc2.weight().q_scale()*16).item()
    
    b_1 = np.round(model.conv1.bias().detach().numpy() / model.quant2.scale.item()).astype(np.int8)
    b_2 = np.round(model.conv2.bias().detach().numpy() / model.quant3.scale.item()).astype(np.int8)
    b_3 = np.round(model.fc1.bias().detach().numpy() / model.quant4.scale.item()).astype(np.int8)
    b_4 = np.round(model.fc2.bias().detach().numpy()).astype(np.int8)


    # print("M values: ", round(M1*(2**16)), round(M2*(2**16)), round(M3*(2**16)))
    # print("b: ", b_1, b_2, b_3)
        
    # print("int_repr")
    w = model.fc1.weight().int_repr().transpose(0, 1).detach().numpy()
    w = w>>4+(w>>3 & 1)
    #print("W: ", w.size, "\n", w)
    w = w.flatten()
    
    
    # print("Compress: ")
    hist, bit = compress_nparr(w, 16)
    #print(hist)
    #print(len(bit))
    fc1_str = exportCArray(bit)
    #print(len(w))
    # +42% increase in performance and 42% decrease in storage
    # Before: (3072000/2/1024*3827)/240000000 ~0.024s
    # After: (893038/1024*3827)/240000000 ~0.014s
    # print("% change: ", (893038-(3072000/2))/(3072000/2))
    # print("% performance: ", (0.014-0.024)/0.024)


    # 3*32*32*128
    # print(fc1_str)

    print("Bias: ")
    bins = 16
    s_b = 2*torch.max(torch.abs(model.fc1.bias()))/bins
    b = Q_int8(model.fc1.bias(), s_b, 0).detach().numpy().astype(np.int8)
    print(b.size)
    #print(model.fc1.bias())

    # generate a .h file that contains: conv1, conv2, fc1, fc2 weights, size, and bias
    f = open("./runtime/esp32-s3/Image-net-conv-net/Image-net-conv-net.h", "w")


    Mstr = "const static uint16_t {} = {};\n"
    f.write(Mstr.format("M1", quantizeInt16(M1)))
    f.write(Mstr.format("M2", quantizeInt16(M2)))
    f.write(Mstr.format("M3", quantizeInt16(M3)))
    f.write(Mstr.format("M4", quantizeInt16(M4)))

    Mstr = "const static uint16_t {} = {};\n"
    f.write(Mstr.format("S1", round(1/model.quant.scale[0].item())))
    f.write(Mstr.format("S2", round(1/model.quant2.scale[0].item())))
    f.write(Mstr.format("S3", round(1/model.quant3.scale[0].item())))
    f.write(Mstr.format("S4", round(1/model.quant4.scale[0].item())))

    arrayStr = "const static int8_t {}[{}] = {};\n"
    f.write(arrayStr.format("conv1_b", len(b_1), exportCArray(b_1)))
    f.write(arrayStr.format("conv2_b", len(b_2), exportCArray(b_2)))
    f.write(arrayStr.format("fc1_b", len(b_3), exportCArray(b_3)))
    f.write(arrayStr.format("fc2_b", len(b_4), exportCArray(b_4)))


    def emitCompress(template, name, layer):
        w = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int8)
        w = w.flatten()
        # print(name)
        w = np.minimum(7, w)
        w = np.maximum(-8, w)
        # print(np.min(w), np.max(w))
        # print(w)
        w += 8
        hist, bit = compress_nparr(w, 16)
        return template.format(name+"_d", len(bit), exportCArray(bit)) + template.format(name+"_f", len(hist), exportCArray(hist))
    
    def emitInt4(template, name, layer):
        w = np.round(layer.weight().dequantize().detach().numpy() / layer.weight().q_scale() / 16).astype(np.int8)        
        w = w.flatten()
        # print(name)
        w = np.minimum(7, w)
        w = np.maximum(-8, w)
        # print(np.min(w), np.max(w))
        # print(w)
        w += 8
        
        arr = np.zeros(len(w)>>1, dtype=np.uint8)
        for i in range(len(arr)):
            arr[i] = (w[i*2] & 15) + (w[(i*2)+1] << 4)
            assert w[i*2] == (arr[i].astype(np.int16) & 15)
            assert w[(i*2)+1] == (arr[i].astype(np.int16) >> 4)
        return template.format(name, len(arr), exportCArray(arr))
    
    arrayStr = "const uint8_t {}[{}] = {};\n"
    f.write(emitCompress(arrayStr, "conv1_w", model.conv1))
    f.write(emitCompress(arrayStr, "conv2_w", model.conv2))
    f.write(emitCompress(arrayStr, "fc1_w", model.fc1))
    f.write(emitCompress(arrayStr, "fc2_w", model.fc2))

    arrayStr = "const uint8_t {}[{}] = {};\n"
    f.write(emitInt4(arrayStr, "conv1_w", model.conv1))
    f.write(emitInt4(arrayStr, "conv2_w", model.conv2))
    f.write(emitInt4(arrayStr, "fc1_w", model.fc1))
    f.write(emitInt4(arrayStr, "fc2_w", model.fc2))
    
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
    
    arrayStr = "const static int8_t {}[{}] = {};\n"
    img = np.round(image.detach().numpy()/0.0078).astype(np.int8).flatten()
    f.write(arrayStr.format("img", len(img), exportCArray(img)))


if __name__ ==  '__main__':
    main()


