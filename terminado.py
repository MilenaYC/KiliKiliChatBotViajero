# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 10:00:30 2021

@author: Milena YC
"""

import nltk
import numpy as np
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

stemming = PorterStemmer()

def tokenize(sentence):
    tokens = word_tokenize(sentence)
    lower_tokens = [t.lower() for t in tokens]
    return lower_tokens

def español(spanish):
    stop_words_sp= set(stopwords.words('spanish'))
    texto=[w for w in spanish if not w in stop_words_sp]
    return texto

def stemm(word):
    return stemming.stem(word)

def bolsa_palabras(tokenize_sentence, todo):
    tokenize_sentence=[stemm(w) for w in tokenize_sentence]
    bag=np.zeros(len(todo), dtype=np.float32)
    for idx, w in enumerate(todo):
        if w in tokenize_sentence:
            bag[idx]=1.0
    return bag
import torch
import torch.nn as nn

class Red(nn.Module):
    def __init__(self, input_size, hidden_size,num_classes):
        super(Red, self).__init__()
        self.l1= nn.Linear(input_size,hidden_size)
        self.l2= nn.Linear(hidden_size,hidden_size)
        self.l3= nn.Linear(hidden_size,num_classes)
        self.activacion = nn.ReLU()
    def forward(self, x):

        out= self.l1(x)
        out= self.activacion(out)
        out= self.l2(x)
        out= self.activacion(out)
        out= self.l3(x)
        
        return out
import json
#from NLTK_uso import tokenize, español, stemm, bolsa_palabras
import numpy as np
nltk.download('punkt')
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


#from modelo import Red

signos=[',',';','(',')','[',']','-','.','?','´','/','{','}','','``',':','=','|','<','>','\'s','\'\'','||','*','|-']

with open('conocimiento.json', 'r') as f:
    conocimiento= json.load(f)

todo=[]
indicadores=[]
xy=[]
for conocer in conocimiento['sabiduria']:
    indicador= conocer['indicador']
    indicadores.append(indicador)
    for entrada in conocer['entradas']:
        aux= tokenize(entrada)
        todo.extend(aux)
        xy.append((aux, indicador))

todo= español(todo)
todo=[stemm(w) for w in todo if w not in signos]
todo= sorted(set(todo))
indicadores= sorted(set(indicadores))

X_train = []
Y_train = []
for (sentence, indicador) in xy:
    bolsa= bolsa_palabras(sentence, todo)
    X_train.append(bolsa)
    
    label= indicadores.index(indicador)
    Y_train.append(label)

X_train = np.array(X_train)
Y_train= np.array(Y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    def __getitem__ (self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    
batch_size=1000
hidden_size=34
output_size= len(indicadores)
input_size= len(X_train[0])
learning_rate= 0.001
num_epochs= 1000

dataset= ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
modelo =Red(input_size, hidden_size, output_size).to(device)

perdida= torch.nn.CrossEntropyLoss()
optimizacion= torch.optim.Adam(modelo.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words= words.to(device)
        labels= labels.to(device)
        
        outputs= modelo(words)

        loss= perdida(outputs, labels.long())
        optimizacion.zero_grad()
        loss.backward()
        optimizacion.step()

    if (epoch +1)%100 ==0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')
print(f'final loss, loss={loss.item():.4f}')

data ={
       "model_state": modelo.state_dict(),
       "input_size": input_size,
       "output_size": output_size,
       "hidden_size": hidden_size,
       "todo": todo,
       "indicadores": indicadores
       }

FILE = "data.pth"
torch.save(data,FILE)

print(f'traning complete. file save to {FILE}')

import random
import json
import torch
#from modelo import Red
#from NLTK_uso import bolsa_palabras, tokenize
import pyttsx3 


device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

with open('conocimiento.json', 'r') as f:
    conocimiento = json.load(f)
    
FILE= "data.pth"
data = torch.load(FILE)

model_state= data["model_state"]
input_size= data["input_size"]
hidden_size= data["hidden_size"]
output_size= data["output_size"]
todo=data["todo"]
indicadores=data["indicadores"]

modelo =Red(input_size, hidden_size, output_size).to(device)
modelo.load_state_dict(model_state)
modelo.eval()

nombre= "KiliKili"
print("¿Cuál es tu proxima aventura?, llévame contigo :) coloca 'salir' para terminar nuestra conversación")
while True:
    sentence= input('Tu:')
    if sentence == "salir":
        print(f"{nombre}: Hasta pronto!!!")
        break
    
    sentence= tokenize(sentence)
    X= bolsa_palabras(sentence, todo)
    X= X.reshape(1, X.shape[0])
    X= torch.from_numpy(X)

    output= modelo(X)

    _, predicted = torch.max(output, dim=1)
    indicador= indicadores[predicted.item()]
    
    proba= torch.softmax(output, dim=1)
    prob= proba[0][predicted.item()]
    #print(prob.item())
    if prob.item() < 0.75:
        for conocer in conocimiento["sabiduria"]:
            if indicador == conocer["indicador"]:
                resp=random.choice(conocer['respuestas'])
                engine = pyttsx3.init()
                engine.say(resp)
                print(f"{nombre}: {resp}")
                engine.runAndWait()
    else:
        print(f"{nombre}: No te logro entender...")
    