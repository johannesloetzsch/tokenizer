import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchinfo import summary
from corpus import Corpus
from embedding.cbow import cbow_trainingdata_for_corpus

c_full = Corpus()
c = Corpus(c_full.corpus[1_500:10_000])
print("Full corpus", c_full.__meta__())
print("Used corpus", c.__meta__())

word_count = len(c.words())
context_size = 2

[X_raw,Y_raw] = cbow_trainingdata_for_corpus(c)
X = F.one_hot(torch.tensor(X_raw), num_classes = word_count).float().reshape((len(X_raw), 2*context_size*word_count))
Y = F.one_hot(torch.tensor(Y_raw), num_classes = word_count).float()  #.reshape((len(Y_raw), word_count))


class Net(nn.Module):
    def __init__(self, in_features, out_features, p_dropout=0.1, **args):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p_dropout)
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

    
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
    print('Train Epoch: {:4d} \tLoss: {:.6f}'.format(epoch, loss.item()))

args = {"epochs": 1000, "batch_size": 32, "p_dropout": 0, "log_interval": 100}

device = torch.device("cpu")
model = Net(2*context_size*word_count, word_count, **args).to(device)
print(summary(model))
optimizer = optim.Adadelta(model.parameters())
train_loader = torch.utils.data.DataLoader(list(zip(X,Y)), batch_size=args["batch_size"])


for epoch in range(1, 1+args["epochs"]):
    train(args, model, device, train_loader, optimizer, epoch)

[idx2word, word2idx] = c.translators()
print(' '.join([idx2word[int(x.argmax())] for x in model(X)]))
