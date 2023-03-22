# This code is using bert-as-service
# server-side program should be run
# clip(bert)-as-service:  https://github.com/jina-ai/clip-as-service

# command line to run server-side bert as service 
# bert-serving-start -model_dir /path/to/bert/model/uncased_L-12_H-768_A-12 -num_worker=8 -max_seq_len=50


# from pytorch_pretrained_bert import BertModel, BertTokenizer
import numpy as np
from tqdm import tqdm
# from tqdm.notebook import tqdm
import torch

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

splitted_dataset = np.load('./dataset/BGL/BGL_full.log_structured_0.8_rad_6.0.npz', allow_pickle=True)
x_train = splitted_dataset["x_train"][()]
y_train = splitted_dataset["y_train"]
x_test = splitted_dataset["x_test"][()]
y_test = splitted_dataset["y_test"]
print(len(x_train))  #dict
print(len(y_train))  #list
print(len(x_test))  #dict
print(len(y_test))  #list

def bert_generator(x_data, s_parse):
    x_data_vec = []
    if not s_parse:
        input_type = "Content"
    else:
        input_type = "EventTemplate"
        
    for blk,seq in tqdm(x_data.items()):
        blk_vec = torch.tensor([], requires_grad=False)
        sens_list = torch.tensor([], requires_grad=False)
        for event in seq:
            text = event[input_type]
            marked_text = "[CLS] " + text + " [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            tokens_tensor = torch.tensor([indexed_tokens])
            segments_tensors = torch.tensor([segments_ids])
            model.eval()
            result = model(tokens_tensor, output_all_encoded_layers=False)[0]
            vec_sen = torch.mean(result,1)
            sens_list = torch.cat((sens_list,vec_sen))
            
        blk_vec = torch.mean(sens_list,0)
        blk_vec_np = blk_vec.detach().numpy()
        x_data_vec.append(blk_vec_np)
    x_data_vec= np.array(x_data_vec)
    return x_data_vec

x_train_feature = bert_generator(x_train, True)
x_test_feature = bert_generator(x_test, True)

np.savez('./dataset/BGL/BGL_BERT_template_0.8_6h.npz',x_train = x_train_feature, y_train = y_train, x_test=x_test_feature, y_test=y_test)