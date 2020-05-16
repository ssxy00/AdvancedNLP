from transformers import BertTokenizer, BertForSequenceClassification
import torch
import random
from torch import optim
import os
from dataset import SCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import BertForSC1,BertForSC2,BertForSC3
#设置参数
batch_size=16
n_epoch=20
device="cuda:1"
checkpoint_path="./checkpoints/Model3"
load_last=False
start_epoch=0
model_file_name="./checkpoints/Model1/checkpoint19.pt"
#设置种子
seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
#准备模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
if load_last:
    model=torch.load(model_file_name).to(device)
else:
    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased').to(device)
    model = BertForSC3.from_pretrained('bert-base-uncased').to(device)
    # model = BertForSC2.from_pretrained('bert-base-uncased').to(device)
    # model = BertForSC1.from_pretrained('bert-base-uncased').to(device)
# optimizer=optim.Adam(model.parameters(),lr=0.0001)
optimizer=optim.Adam(model.classifier.parameters(),lr=0.0001)
#准备数据集
train_dir_name=os.path.join(os.getcwd(),"Task_1","Data","train_and_dev","task_1","train")
dev_dir_name=os.path.join(os.getcwd(),"Task_1","Data","train_and_dev","task_1","dev")
print("prepare train dataset")
train_dataset=SCDataset(train_dir_name,tokenizer)
print("prepare valid dataset")
dev_dataset=SCDataset(dev_dir_name,tokenizer)
train_dataloader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=train_dataset.collate_func,num_workers=2)
dev_dataloader=DataLoader(dev_dataset,batch_size=batch_size,shuffle=True,collate_fn=dev_dataset.collate_func,num_workers=2)
torch.cuda.empty_cache()
#模型训练
for i_epoch in range(n_epoch):
    epoch=start_epoch+i_epoch
    print("train epoch ",epoch)
    model.train()
    tqdm_data=tqdm(train_dataloader,desc='Train(epoch {})'.format(epoch))
    for i,data in enumerate(tqdm_data):
        outputs = model(data['src'].to(device), labels=data['tgt'].to(device),attention_mask=data['attention_mask'].to(device))
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        model.zero_grad()
        tqdm_data.set_postfix({'loss':loss})
    print("valid epoch ",epoch)
    with torch.no_grad():
        model.eval()
        tqdm_data=tqdm(dev_dataloader,desc='valid(epoch {})'.format(epoch))
        average_loss=0
        average_acc=0
        for i,data in enumerate(tqdm_data):
            outputs = model(data['src'].to(device), labels=data['tgt'].to(device),attention_mask=data['attention_mask'].to(device))
            loss = outputs[0]
            pre=torch.tensor([1 if i[1]>i[0] else 0 for i in outputs[1]]).to(device)
            acc=float(torch.sum(pre==data['tgt'].to(device).squeeze(1)))/batch_size
            average_loss=(average_loss*i+loss.item())/(i+1)
            average_acc=(average_acc*i+acc)/(i+1)
            tqdm_data.set_postfix({'loss':loss.item(),'acc':acc})
        print(epoch,"'s ave_dev_loss is :",average_loss)
        print(epoch,"'s ave_dev_acc is :",average_acc)
    torch.save(model,os.path.join(checkpoint_path,'checkpoint{}.pt'.format(epoch)))
    torch.cuda.empty_cache()