import torch
from numpy import *
import matplotlib.pyplot as plt
import torchvision.models as models
from NumbersData import NumbersData
from torch.utils.data import DataLoader
from NeuralNetwork import NeuralNetwork
from torchvision.transforms import ConvertImageDtype, ToTensor


class NumbersNN:
    
    def __init__(self):
        self.__model = NeuralNetwork()


    def create_model(self, name: str):
        torch.save(self.__model, f"{name}.pth")


    def load_model(self, model_path: str):
        self.__model = torch.load(model_path)


    def unload_model(self):
        self.__model = NeuralNetwork()


    def train_model(self, csv_file_path: str, images_dir_path: str, save_model_path: str = "model.pth", epochs: int = 1):

        batch_size = 64
        learning_rate = 1e-3

        train_data = NumbersData(csv_file_path, images_dir_path, transform=ConvertImageDtype(torch.float).forward)
        self.train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.__model.parameters(), lr=learning_rate)

        # Обучение и сохранение модели
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            self.__train_loop()
        print("Done!")
        if input("save? (y/n) ") == 'y':
            torch.save(self.__model, save_model_path)


    def __train_loop(self):
        
        size = len(self.train_dataloader.dataset)
        num_batches = len(self.train_dataloader)
        test_loss, correct = 0, 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            pred = self.__model(X)
            loss = self.loss_fn(pred, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    def check_number(self, l, lst):
        image = torch.tensor(l, dtype=torch.uint8)
        image = ConvertImageDtype(torch.float).forward(image)
        l = [ 0 if i < 0 else i for i in self.__model(image).tolist()[0]]
        for i in range(10):
            lst[i]['value'] = l[i] / sum(l) * 100 if l[i] > 0 else 0

# b = AINumers()
# b.load_model("model.pth")
# b.train_model("numbers_labels.csv", "train")


'''
l = [[[[1, 2, 3],
    [2, 3, 4],
    [4, 6, 2]]]]
'''
# def check_number(l, lst):
#     model = NeuralNetwork()
#     model = torch.load("model.pth")
#     image = torch.tensor(l, dtype=torch.uint8)
#     image = ConvertImageDtype(torch.float).forward(image)
#     # print(model(image))
#     # print(image)
#     l = [ 0 if i < 0 else i for i in model(image).tolist()[0]]
#     for i in range(10):
#         lst[i]['value'] = l[i] / sum(l) * 100 if l[i] > 0 else 0
    
