import os
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import CountVectorizer
import json

script_dir = os.path.dirname(os.path.abspath(__file__))

config_file_path = os.path.join(script_dir, "model_config.json")
with open(config_file_path, "r") as json_file:
    config_data = json.load(json_file)

max_features = config_data["max_features"]

vectorizer = CountVectorizer(max_features=max_features)

vocabulary_file_path = os.path.join(script_dir, 'vectorizer_vocabulary.json')
with open(vocabulary_file_path, 'r') as f:
    vocabulary = json.load(f)

vectorizer.vocabulary_ = vocabulary

class SentimentClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SentimentClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        x = self.fc(x)
        return x

def preprocess_text(text, vectorizer):
    if text:
        text_vector = vectorizer.transform([text])
        return torch.tensor(text_vector.toarray(), dtype=torch.float32)
    else:
        return None

def predict_sentiment(text, model, vectorizer):
    if text:
        model.eval()
        with torch.no_grad():
            text_tensor = preprocess_text(text, vectorizer)
            if text_tensor is not None:
                output = model(text_tensor)
                _, predicted = torch.max(output, 1)
                sentiment = predicted.item()
                return sentiment
    return None

model = SentimentClassifier(max_features, 3)

model_file_path = os.path.join(script_dir, "model.pth")

model_state_dict = torch.load(model_file_path, map_location=torch.device('cpu'))

state_dict = model_state_dict['model_state_dict']

model.load_state_dict(state_dict, strict=False)

while True:
    user_input = input("Enter a comment (press Enter to exit): ")
    if user_input.strip() == "":
        break
    sentiment = predict_sentiment(user_input, model, vectorizer)
    if sentiment is not None:
        if sentiment == 0:
            print("Negative")
        elif sentiment == 1:
            print("Neutral")
        else:
            print("Positive")
