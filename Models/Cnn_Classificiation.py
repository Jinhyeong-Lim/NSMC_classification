import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from konlpy.tag import Okt
from torchtext import data

okt = Okt()

ID = data.Field(sequential = False, # 순착적인지 나타내는 여부
                use_vocab = False) # 실제로 사용 x

TEXT = data.Field(sequential=True,
                  use_vocab=True,
                  tokenize=okt.morphs,
                  lower=True,
                  batch_first=True,
                  fix_length=20) # 모든 text length를 fix_length에 맞추고 길이가 부족하면  padding

LABEL = data.Field(sequential=False,
                   use_vocab=False,
                   is_target=True) # 대상 변수인지 여부,

from torchtext.data import TabularDataset

train_data, test_data = TabularDataset.splits(
        path='.', train='ratings_train.txt', test='ratings_test.txt', format='tsv',
        fields=[('id', ID), ('text', TEXT), ('label', LABEL)], skip_header=True)

print(vars(train_data[0]))
#counter
TEXT.build_vocab(train_data, min_freq=10, max_size=70000) # 최소 10번 이상 나온 단어 word2index
print(TEXT.vocab.stoi)

from torchtext.data import Iterator
batch_size = 50
train_loader = Iterator(dataset=train_data, batch_size = batch_size, shuffle=True)
test_loader = Iterator(dataset=test_data, batch_size = batch_size)

class CNN(nn.Module):
    def __init__(self, batch ,vocab_size, length ,input_size): #(batch_size,출현 단어 개수, 1개 문장 tensor 길이, embedding_dim)
        super(CNN, self).__init__()
        self.input_size = input_size #embedding_dim
        self.batch = batch #batch_size(50)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size,  # 워드 임베딩 ()
                                            embedding_dim=input_size)

        self.conv3 = nn.Conv2d(1, 100, (3, input_size),bias=True) # kernerl_size(3*input_size), ouput_channel=1
        self.conv4 = nn.Conv2d(1, 100, (4, input_size),bias=True) # kernerl_size(4*input_size), ouput_channel=1
        self.conv5 = nn.Conv2d(1, 100, (5, input_size),bias=True) # kernerl_size(5*input_size), ouput_channel=1
        self.Max3_pool = nn.MaxPool2d((length - 3 + 1, 1)) #pooling(length-3+1)값 중 max 값
        self.Max4_pool = nn.MaxPool2d((length - 4 + 1, 1)) #pooling(length-4+1)값 중 max 값
        self.Max5_pool = nn.MaxPool2d((length - 5 + 1, 1)) #pooling(length-5+1)값 중 max 값
        self.linear1 = nn.Linear(300, 1) # Fully_connected
        self.dropout = nn.Dropout(p=0.5) #Dropout(Regularization)

    def forward(self, x):
        # 1. 임베딩 층
        # 크기변화: (배치 크기, 시퀀스 길이) => (배치 크기, 시퀀스 길이, 임베딩 차원)

        output = self.embedding_layer(x)
        x = output.view(self.batch, 1, -1, self.input_size)  # batchsize, channel, seq_len, embed_size
        x1 = F.relu(self.conv3(x)) # batchsize, output_channel, (length-3+1, 1) feature map
        x2 = F.relu(self.conv4(x)) # batchsize, output_channel, (length-4+1, 1) feature map
        x3 = F.relu(self.conv5(x)) # batchsize, output_channel, (length-5+1, 1) feature map
        # Pooling
        x1 = F.relu(self.Max3_pool(x1)) # batchsize, output_channel,1,1
        x2 = F.relu(self.Max4_pool(x2)) # batchsize, output_channel,1,1
        x3 = F.relu(self.Max5_pool(x3)) # batchsize, output_channel,1,1
        # capture and concatenate the features
        x = torch.cat((x1, x2, x3), -1) #batchsize, output_channel(100), 1,3
        #print(x.shape)
        x = x.view(self.batch, 300) #batchsize, 3
        x = self.dropout(x)
        # project the features to the labels
        x = F.sigmoid(self.linear1(x)) #batchsize,1

        return x


net = CNN(50, len(TEXT.vocab), 20, 20) # batch_size, 총단어 개수, pad_sequence문장길이,)
                                          # embeding dimension 개수


criterion = nn.BCELoss()
optimizer = optim.RMSprop(net.parameters(), lr =0.001, weight_decay=1e-3)

losses=[]
acc=[]
for epoch in range(20):
    for batch in train_loader:
        optimizer.zero_grad()
        #print(batch.text)
        predictions = net(batch.text)
        #print(hypothesis)
        loss = criterion(predictions, batch.label.float())
        #print(cost)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f'Epoch {epoch + 1},training loss: {torch.tensor(losses).mean()}')

test_losses = []  # track loss
num_correct = 0

net.eval() #test mode로 변경, dropout같은 함수가 적용 될지 안될지 결정
# iterate over test data
for batch in test_loader:
    #batch_size=50000
    # get predicted outputs
    output = net(batch.text)
    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(batch.label.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy())
    num_correct += np.sum(correct)

test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))