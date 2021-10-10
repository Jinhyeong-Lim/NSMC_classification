import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import ElectraForSequenceClassification
from transformers import ElectraTokenizer


train_df = pd.read_csv('ratings_train1.txt', sep='\t')
test_df = pd.read_csv('ratings_test.txt', sep='\t')
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
test_df = test_df.sample(frac=0.1, random_state=999)


class NsmcDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.df.iloc[idx, 1]
        label = self.df.iloc[idx, 2]
        return text, label


nsmc_train_dataset = NsmcDataset(train_df)
train_loader = DataLoader(nsmc_train_dataset, batch_size=1, shuffle=True, num_workers=2)
nsmc_eval_dataset = NsmcDataset(test_df)
test_loader = DataLoader(nsmc_eval_dataset, batch_size=1, shuffle=False, num_workers=2)

tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 1
total_loss = 0
total_len = 0
total_correct = 0

model.train()
for epoch in range(epochs):
    for text, label in train_loader:
        optimizer.zero_grad()
        encoded_list = [tokenizer.encode_plus(t, add_special_tokens=True, max_length = 512, pad_to_max_length = True)['input_ids'] for t in text]
        encoded_list = torch.tensor(encoded_list)
        sample, label = encoded_list.to(device), label.to(device)
        labels = torch.tensor(label)
        outputs = model(sample, labels=labels)
        # print(outputs)
        loss = outputs.loss
        logits = outputs.logits

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch + 1, epochs, itr,
                                                                                              total_loss / p_itr,
                                                                                              total_correct / total_len))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr += 1

model.eval()
total_loss = 0
total_len = 0
total_correct = 0
with torch.no_grad():
    for text, label in test_loader:
        encoded_list = [tokenizer.encode_plus(t, add_special_tokens=True, max_length = 512, pad_to_max_length = True)['input_ids'] for t in text]

        sample = torch.tensor(encoded_list)
        labels = torch.tensor(label)
        sample, labels = sample.to(device), labels.to(device)
        outputs = model(sample)
        loss = outputs.loss
        logits = outputs.logits

        pred = torch.argmax(F.softmax(logits), dim=1)
        correct = pred.eq(labels)
        total_correct += correct.sum().item()
        total_len += len(labels)

    print('Test accuracy: ', total_correct / total_len)