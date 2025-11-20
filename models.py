import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# 1. Linear Baseline
class LinearModel:
    def __init__(self):
        self.model = LogisticRegression()
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

# 2. Kernel Machine (SVM)
class KernelModel:
    def __init__(self, gamma='scale', C=1.0):
        self.model = SVC(kernel='rbf', gamma=gamma, C=C)
        
    def fit(self, X, y):
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

# 3. Wide Shallow Net
class ShallowNet(nn.Module):
    def __init__(self):
        super(ShallowNet, self).__init__()
        self.fc1 = nn.Linear(2, 1000)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1000, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# 4. Deep Net
class DeepNet(nn.Module):
    def __init__(self):
        super(DeepNet, self).__init__()
        self.fc1 = nn.Linear(2, 20)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(20, 20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 20)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(20, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        return out

def train_torch_model(model, X, y, epochs=1000, lr=0.01, callback=None):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).unsqueeze(1)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
            
        if callback and (epoch % 10 == 0 or epoch == epochs - 1):
             callback(model, epoch)
            
    return model
