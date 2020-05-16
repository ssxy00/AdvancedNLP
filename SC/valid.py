from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
from Task_1.Evaluation_.eval_1 import task_1_eval_main

class validation(object):
    def __init__(self,model_name,tokenizer,device):
        self.fname=[]
        self.model=self.get_model(model_name).to(device)
        self.tokenizer=tokenizer
        self.device=device

    def append_dir(self,dir_name):
        for i in os.listdir(dir_name):
            self.fname.append(os.path.join(dir_name,i))

    def append_fname(self,fname):
        self.fname.append(fname)

    def get_model(self,model_name):
        print("loading model ..........")
        return torch.load(model_name)

    def generate(self,save_dir=os.getcwd(),hasTarget=False):
        #根据添加的文件列表生成对应的预测
        #hasTarget 表示文件是否是一个句子后面一个标签的形式
        #hasTarget=True ,输入类似dev里面的文件
        #hasTarget=False，输入类似test里面的文件
        print("genetating files ..........")
        self.model.eval()
        for file_name in self.fname:
            print(file_name,"are progressing")
            with open(file_name,"r",encoding="utf-8") as f:
                context=f.readlines()
                targets=[]
                sources=[]
                #每行进行预测
                for line in context:
                    if hasTarget:
                        if line[0].isdigit():
                            n1=line.find('.')
                        else:
                            n1=1
                        n2=line.find('\t')
                        src=line[n1+1:n2-1]
                        source=line[:n2]
                    else:
                        if line[0].isdigit():
                            n1=line.find('.')
                        else:
                            n1=1
                        src=line[n1:-1]
                        source=line
                    sources.append(source)
                    source=torch.tensor(self.tokenizer.encode(src, add_special_tokens=True)).unsqueeze(0)
                    logits=self.model(source.to(self.device))[0]
                    target=torch.argmax(logits)
                    targets.append(str(target.item()))
                #写入新文件
            fname=os.path.split(file_name)[-1]
            new_fname=os.path.join(save_dir,fname.split('.')[0]+"_pred.deft")
            with open(new_fname,"w") as f:
                for i,source in enumerate(sources):
                    f.write(source.strip()+"\t"+targets[i]+"\n")
            print("saved in ",new_fname)
            if hasTarget:
                self.F_1Score(file_name,new_fname)

    def F_1Score(self,gold_fname,pred_fname):
        eval_labels=["0","1"]
        report=task_1_eval_main(gold_fname, pred_fname, eval_labels)
        print(report)



if __name__ == '__main__':
    model_name="checkpoints/Model2/checkpoint24.pt"
    tokenizer=BertTokenizer.from_pretrained('bert-base-uncased')
    device='cuda:2'
    v=validation(model_name,tokenizer,device)
    # v.append_fname("../Task_1/Data/train_and_dev/task_1/dev/task_1_t1_biology_0_303.deft")
    # v.append_dir("/Task_1/Data/train_and_dev/task_1/dev")
    v.append_dir("Task_1/Data/test/subtask_1")
    # v.generate(save_dir="../valid",hasTarget=True)
    v.generate(save_dir="./valid",hasTarget=False)