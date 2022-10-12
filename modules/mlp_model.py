import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
random_seed = 1388
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
#cudnn.benchmark = True       
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):
    def __init__(self, x_data, label):
        self.data = x_data.values.tolist()
        self.label = label.values.tolist()
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return torch.from_numpy(np.array(data)).float(), torch.from_numpy(np.array(label)).float()

    def __len__(self):
        return len(self.label)


# 网络结构
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.fc1 = nn.Linear(input_size, 800)
        self.bn1 = nn.BatchNorm1d(800)
        self.fc2 = nn.Linear(800, 600)
        self.bn2 = nn.BatchNorm1d(600)
        self.fc3 = nn.Linear(600, 300)
        self.fc4 = nn.Linear(300, 1)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.fc1(x)
        y1 = self.bn1(y1)
        # y1 = self.dropout(y1)
        y2 = self.relu(self.fc2(y1))
        y2 = self.bn2(y2)
        y3 = self.relu(self.fc3(y2))
        out = self.fc4(y3)
        # out = self.sigmoid(y4)
        return out
    
class MLPRegression(object):
    def __init__(self):
        pass

    def train(self, x_train,y_train, x_test, y_test):
        # 超参数
        learning_rate = 5e-4
        input_size = x_train.shape[1]
        num_epoches = 50
        batch_size = 64
        hidden_size = 64
        num_layers = 4

        # 训练集数据
        train_data = MyDataset(x_train,y_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        # 测试集数据
        test_data = MyDataset(x_test,y_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        
        # 模型
        mlp = MLP(input_size, hidden_size, num_layers).to(device)   

        # 定义损失函数和优化器
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.Adam(mlp.parameters(), lr=learning_rate)

        # 迭代训练
        best_score = 0
        for epoch in range(num_epoches):
            total_loss = 0
            mlp.train()
            for i, (x, label) in enumerate(train_loader):
                x = x.to(device)
                label = label.to(device)
                outputs = mlp(x)          # 前向传播
                logits = outputs.view(-1)           # 将输出展平
                loss = criterion(logits, label)    # loss计算
                total_loss += loss
                optimizer.zero_grad()               # 梯度清零
                loss.backward(retain_graph=True)    # 反向传播，计算梯度
                optimizer.step()                    # 梯度更新
            # print("epoch:{}, train loss:{}".format(epoch+1, total_loss))
           
            # test
            y_pred, y_true = [], []
            mlp.eval()
            with torch.no_grad():
                for x, label in test_loader:
                    x = x.to(device)
                    outputs = mlp(x)         # 前向传播
                    outputs = outputs.view(-1)          # 将输出展平
                    y_pred.append(outputs)
                    y_true.append(label)
            y_prob = torch.cat(y_pred).cpu()
            y_true = torch.cat(y_true).cpu()
            range_accuracy_100 = range_accuracy(y_true,y_prob,range_pct=0.1,is_log_trans=False)
            # print('range 10% accuracy',range_accuracy_100)
            if range_accuracy_100>best_score:
                best_score = max(best_score, range_accuracy_100)
                model_path = "./model/mlp.model"
                torch.save(mlp, model_path)
                # print("epoch {}, best score is {}(range 10% accuracy),saved best model at:{} ".format(epoch, best_score, model_path))

    def pred(self, x_data):
        # 超参数
        input_size = x_data.shape[1]
        hidden_size = 64
        num_layers = 4
        mlp = MLP(input_size, hidden_size, num_layers).to(device)
        mlp = torch.load('./model/mlp.model')
        x_data = torch.from_numpy(x_data.values).float()
        y_pred = []
        mlp.eval()
        with torch.no_grad():
            for x in x_data:
                x = x.unsqueeze(0)
                x = x.to(device)
                outputs = mlp(x)         # 前向传播
                outputs = outputs.view(-1)          # 将输出展平
                y_pred.append(outputs)
        y_prob = torch.cat(y_pred).cpu()

        return y_prob
