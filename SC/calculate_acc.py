from transformers import BertTokenizer
import torch
import random
import os
from dataset import SCDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
#设置参数
start_epoch=5
end_epoch=20
batch_size=8
device="cuda:2"
model_dir_name="checkpoints/Froze/"
#设置种子
seed=0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
#准备模型
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#准备数据集
dev_dir_name=os.path.join(os.getcwd(),"Task_1","Data","train_and_dev","task_1","dev")
print("prepare valid dataset")
dev_dataset=SCDataset(dev_dir_name,tokenizer)
dev_dataloader=DataLoader(dev_dataset,batch_size=batch_size,shuffle=True,collate_fn=dev_dataset.collate_func,num_workers=2)
torch.cuda.empty_cache()
#模型检测
print("begin to valuate")
for epoch in range(start_epoch,end_epoch):
    model_file_name=model_dir_name+"checkpoint"+str(epoch)+".pt"
    print("weight loaded in ",model_file_name)
    model=torch.load(model_file_name).to(device)
    with torch.no_grad():
        model.eval()
        tqdm_data=tqdm(dev_dataloader,desc='valid()')
        average_acc=0
        TP=0# True positive
        TN=0 # True Negetive
        FN=0 # False Negetive
        for i,data in enumerate(tqdm_data):
            outputs = model(data['src'].to(device),attention_mask=data['attention_mask'].to(device))
            pre=torch.tensor([1 if i[1]>i[0] else 0 for i in outputs[0]]).to(device).bool()
            # acc=float(torch.sum(pre==data['tgt'].to(device).squeeze(1)))/batch_size
            # average_acc=(average_acc*i+acc)/(i+1)
            # tqdm_data.set_postfix({'acc':acc})
            label=data['tgt'].to(device).squeeze(1).bool()
            tmp1=(pre==label)
            TP+= float(torch.sum(torch.tensor([1 if tmp1[i]&label[i] else 0 for i in range(len(label))])))
            TN+= float(torch.sum(torch.tensor([1 if pre[i]& (not tmp1[i]) else 0 for i in range(len(label))])))
            FN+= float(torch.sum(torch.tensor([1 if (not pre[i])& (not tmp1[i]) else 0 for i in range(len(label))])))
        precision=TP/(TP+TN)
        recall=TP/(TP+FN)
        F1=2*TP/(2*TP+TN+FN)
        print("precision :", precision,"\t", "recall: ",recall,"\t","F1: ",F1,"\n")