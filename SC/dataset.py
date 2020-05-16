import os
from torch.utils.data import Dataset
import torch

class SCDataset(Dataset):
    def __init__(self,path,tokenizer):
        super(SCDataset,self).__init__()
        self.tokenizer=tokenizer
        self.path=path
        data=self.parse_data(self.path)
        self.src=data[0]
        self.tgt=data[1]

    def parse_data(self,path):
        source=[]
        target=[]
        for i in os.listdir(path):
            with open(os.path.join(path,i),"r",encoding="utf-8") as f:
                context=f.readlines()
                for line in context:
                    if line[2].isdigit():
                        n1=line.find('.')
                    else:
                        n1=1
                    n2=line.find('\t')
                    src=line[n1+1:n2-1]
                    source.append(self.tokenizer.encode(src, add_special_tokens=True))
                    tgt=line[n2+2]
                    target.append(int(tgt))
        return source,target

    def __getitem__(self, idx):
        return self.src[idx],[self.tgt[idx]]

    def __len__(self):
        return len(self.src)

    def collate_func(self,data):
        instance={"src":[],"tgt":[]}
        for example in data:
            instance["src"].append(example[0])
            instance["tgt"].append(example[1])
        # padding
        key="src"
        instance[key] = [torch.tensor(d, dtype=torch.long) for d in instance[key]]
        instance[key],instance['attention_mask'] = pad_sequence(instance[key], batch_first=True, padding_value=self.tokenizer.pad_token_id)
        instance['tgt']=torch.tensor(instance['tgt'])
        return instance


def pad_sequence(sequences, batch_first=False, padding_value=0):
    # assuming trailing dimensions and type of all the Tensors
    # in sequences are same and fetching those from sequences[0]
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max([s.size(0) for s in sequences])
    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims
    attention_mask=torch.zeros(len(sequences),max_len,dtype=torch.uint8)
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            attention_mask[i,:length,...]=1
            out_tensor[i, :length, ...] = tensor
        else:
            attention_mask[:length,i,...]=1
            out_tensor[:length, i, ...] = tensor

    return out_tensor,attention_mask


