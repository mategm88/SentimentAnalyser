import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import numpy as np
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

reddit_data_file_path = os.path.join(script_dir, "Reddit_Data.csv")
redditData = pd.read_csv(reddit_data_file_path)

twitter_data_file_path = os.path.join(script_dir, "Twitter_Data.csv")
data2 = pd.read_csv(twitter_data_file_path)

PATH = os.path.join(script_dir, "model.pth")

X = redditData['clean_comment']
y = redditData['category']
X2 = data2['clean_text']
y2 = data2['category']

X.fillna('', inplace=True)
X2.fillna('', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train = pd.concat([X_train, X2], ignore_index=True)
y_train = pd.concat([y_train, y2], ignore_index=True)


label_map = {-1: 0, 0: 1, 1: 2}
y_train_mapped = y_train.map(label_map)
y_test_mapped = y_test.map(label_map)


vectorizer = CountVectorizer(max_features=25000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# PyTorch Dataset
class SentimentDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X.toarray(), dtype=torch.float32)
        self.y = torch.tensor(y.values, dtype=torch.long)
        self.length = len(self.X)
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx >= self.length:
            raise IndexError(f"Index {idx} is out of bounds for dimension 0 with size {self.length}")
        return self.X[idx], self.y[idx]



def filter_data(X, y):
    filtered_indices = [i for i, label in enumerate(y) if not pd.isnull(label)]
    return X[filtered_indices], y.iloc[filtered_indices]

X_train_vec_filtered, y_train_mapped_filtered = filter_data(X_train_vec, y_train_mapped)
X_test_vec_filtered, y_test_mapped_filtered = filter_data(X_test_vec, y_test_mapped)


train_dataset = SentimentDataset(X_train_vec_filtered, y_train_mapped_filtered)
test_dataset = SentimentDataset(X_test_vec_filtered, y_test_mapped_filtered)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model
class SentimentClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        return x

input_size = X_train_vec.shape[1]
output_size = len(y.unique())
model = SentimentClassifier(input_size, output_size)

data = {
    "input_size": len(train_dataset),
    "output_size": output_size,
    "max_features": vectorizer.max_features
}

json_file_path = os.path.join(script_dir, "model_config.json")

with open(json_file_path, "w") as json_file:
    json.dump(data, json_file)

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, train_loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

train(model, train_loader, criterion, optimizer)


torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'criterion_state_dict': criterion.state_dict()
}, PATH)

# Evaluation
def evaluate(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")

evaluate(model, test_loader)


vocab_dict = {str(key): int(value) for key, value in vectorizer.vocabulary_.items()}
vocab_file_path = os.path.join(script_dir, 'vectorizer_vocabulary.json')
with open(vocab_file_path, 'w') as f:
    json.dump(vocab_dict, f)
