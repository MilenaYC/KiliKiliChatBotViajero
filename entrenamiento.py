import json
from NLTK_uso import tokenize, español, stemm, bolsa_palabras
import numpy as np
import nltk
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
nltk.download('punkt')


from modelo import Red

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
hidden_size=37
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









