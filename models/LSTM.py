# Instruction:
# 1. load the representations and corresponding labels.
# 2. 'torchsampler' is used to generate balanced training samples(pos/neg) to facilitate the training process
# 3. collate_fn function generates sliding windows and is revoked automatically by the dataloader. Parameters for sliding window can be set with 'log_seq_len' -> window size, 'stride' -> stride

log_seq_len = 50
emb_size = train_set[0][0].shape[1]
# stride = 50
stride = log_seq_len
import os
import h5py

path = "./dataset"

dataset = "FastText"


path_dic ={
    "Word2Vec": {
        "train": "word2vec_x_train_feature_template_300d_mean.npy",
        "test": "word2vec_x_test_feature_template_300d_mean.npy"
    },
    "FastText": {
        "train": "BGL_fasttext_template_tfidf_50d_0.8_6h.npz"
    },
    "BERT": {
        "train": "BGL_bert_x_train_feature_template_768d.npy",
        "test": "BGL_bert_x_test_feature_template_768d.npy"
    },
    "TFIDF": {
        "train": "HDFS_TFIDF_Text_Non_Win.npz",
        "test": "HDFS_TFIDF_Text_Non_Win.npz"
    }
}


def path_to_data(dataset):
    train_file = os.path.join(path, path_dic[dataset]['train'])
    test_file = os.path.join(path, path_dic[dataset]['test'])
    return {'train':train_file, 'test':test_file}


import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support
import warnings
warnings.filterwarnings("ignore")


from functools import partial


import torch
import sys,time,math
from torch import nn, optim
from torch.autograd import Variable
from torchvision import transforms
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchsampler import ImbalancedDatasetSampler


from random import uniform
from IPython.display import display, clear_output
from ipywidgets import IntSlider, Output

import matplotlib.pyplot as plt


from tqdm import tqdm


class NPYDataset(Dataset):
    """Dataset wrapping data and target tensors.
    Arguments:
    """

#     def __init__(self, x_data_path, y_dataset):
#         self.x_data = np.load(x_data_path, allow_pickle=True)
#         self.y_data = y_dataset

    def __init__(self, x_dataset, y_dataset):
        self.x_data = x_dataset
        self.y_data = y_dataset

    
    def __getitem__(self, index):
        # print(index)
        if not hasattr(self, 'y_data'):
            return self.x_data[index]
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)
    
    def get_labels(self):
        return self.y_data
    

dataset = np.load(path_to_data(dataset)['train'],allow_pickle=True)
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

train_set = NPYDataset(x_train, y_train)
test_set = NPYDataset(x_test, y_test)

def collate_fn(batch, emb_size, win_size=10, stride = 1):
    ret = []
    labels = []
#     padding_emb = torch.zeros(emb_size).reshape(1,-1)
    padding_emb = np.zeros(emb_size).reshape(1,-1)
#     print(padding_emb.shape)
    for session in batch:
        fea, label = session
        ses_len = fea.shape[0]
        # sliding window
#         print(ses_len)
        if win_size>ses_len:
            rem = np.array(fea[fea.shape[0]-win_size+1:])
            while len(rem)<win_size:
                rem = np.append(rem, padding_emb, axis = 0)
            splits = np.expand_dims(rem, axis=0)
#             print("small:",splits.shape)
        else:
            splits = np.array([fea[i:i+win_size] for i in range(0, fea.shape[0]-win_size+1, stride)])
            # append the last one
            l_start = range(0, fea.shape[0]-win_size+1, stride)[-1]
            last_one = fea[l_start+stride:]
            if len(last_one)>0:
                while len(last_one)<win_size:
                    last_one = np.append(last_one, padding_emb, axis = 0)
                last_one = np.expand_dims(last_one, axis=0)
                splits = np.append(splits, last_one, axis = 0)
        ret.append(splits)
        labels.append(label)
    return {'x': ret,  'y': labels}

log_seq_len = 50
emb_size = train_set[0][0].shape[1]
# stride = 50
stride = log_seq_len
batch_size = 128
# batch_size = 100
# learning_rate = 1e-3
# num_epoches = 20

train_loader = DataLoader(dataset=train_set, sampler=ImbalancedDatasetSampler(train_set), num_workers=2, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, win_size=log_seq_len, stride = stride, emb_size = emb_size))
test_loader = DataLoader(dataset=test_set, num_workers=2, batch_size=batch_size, shuffle=False, collate_fn=partial(collate_fn, win_size=log_seq_len, stride = stride, emb_size = emb_size)) 

# RNN Model

class Rnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(Rnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer,
                            batch_first=True)
        self.classifier = nn.Linear(hidden_dim, n_class)
        self.model_name = 'LSTM'

    def forward(self, x):
        # h0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        # c0 = Variable(torch.zeros(self.n_layer, x.size(1),
                                #   self.hidden_dim)).cuda()
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.classifier(out)
        return out

    def train(self, train_loader, test_loader, num_epoches=100, eval_interval=1, lr = 0.01):
        best_f1 = 0
        b_prec = 0
        b_rec =0
        b_epoch = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        for epoch in range(num_epoches):
            print('epoch {}'.format(epoch + 1))
            print('*' * 10)

            running_loss = 0.0
            running_acc = 0.0

            output = Output()
            display(output)
            cnt=0
#             dataset_len= len(train_loader)

            for i_s, data in enumerate(tqdm(train_loader)):
                samples = data['x']
                labels = data['y']
                pred = []
                i_loss = 0
                for i, sample in enumerate(samples):
                    label = torch.tensor(np.repeat(labels[i], len(sample)),dtype = torch.long)
                    sample = sample.astype(float)
                    sample = torch.from_numpy(sample)
                    out = self.forward(sample.float())
                    loss = criterion(out, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    i_loss = i_loss + loss.data.item()

                    _, i_pred = torch.max(out, 1)
                    pred.append(1 if i_pred.sum()>0 else 0)
#                 print('pred', pred)
#                 print('labels',labels)
                num_correct = (np.array(pred) == labels).sum()
                cnt=cnt+len(labels)
                running_loss += i_loss * len(labels)
                running_acc += num_correct


                with output:
                    clear_output(wait=True)
                    display('Batch: acc: {:.3f} loss: {:.3f} running: acc: {:.3f}, loss: {:.3f}'.format(num_correct.item()/len(labels), i_loss, running_acc/cnt, running_loss/cnt))


            if epoch % eval_interval == 0:
                print("Evaluation")
                print('*' * 10)
                precision, recall, f1 = self.evaluation(test_loader) 
                if f1 > best_f1:
                    best_f1 = f1
                    b_prec = precision
                    b_rec = recall
                    b_epoch = epoch + 1
                    print('Saving model to the disk...')
                    torch.save(self, os.path.join(path, dataset+'_'+self.model_name+'.pkl'))
                    
            print('Best Epoch: {:d}, Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(b_epoch, b_prec, b_rec, best_f1))

        return {'b_epoch': b_epoch,
                'precision': b_prec,
                'recall': b_rec,
                'f1': best_f1}


    def evaluation(self, test_loader):
        # Evaluation window_based calculation
        prediction_list = []
        label_list = []
        for data in tqdm(test_loader):
            samples = data['x']
            labels = data['y']
            pred = []
            labels_broadcasting = []
            for i, sample in enumerate(samples):
                sample = sample.astype(float)
                sample = torch.from_numpy(sample)
                len_win = sample.shape[0]
                sample = sample.unsqueeze(1)
                out = self.forward(sample.float())
                _, i_pred = torch.max(out, 1)
                pred = pred+i_pred.tolist()
                labels_broadcasting = labels_broadcasting + [labels[i]]*len_win
                
            prediction_list = prediction_list+pred
            label_list = label_list + labels_broadcasting

        precision, recall, f1, _ = precision_recall_fscore_support(label_list, prediction_list, average='binary')
        
        print('Testset: Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        
        return precision, recall, f1
        
LSTM_model = Rnn(emb_size, 8, 1, 2)  # input dim, hidden dim, n_layer, output dim
LSTM_model.float()
best_result = LSTM_model.train(train_loader, test_loader, num_epoches = 30, eval_interval = 1, lr = 0.001)