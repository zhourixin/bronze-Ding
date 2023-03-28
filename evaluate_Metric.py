import torch.hub
from sklearn.metrics import average_precision_score
import os
import random
import numpy as np
import csv



os.environ["CUDA_VISIBLE_DEVICES"] = '0'
seed = 2022
np.random.seed(seed)
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.benchmark = False 
torch.backends.cudnn.deterministic = True 
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)



def get_onehot(targets, classnum=11):
    target_list = []
    for i in range(targets.size(0)):
        state = [0]*classnum        
        level_label = int(targets[i])
        state[level_label] = 1
        state = torch.from_numpy(np.asarray(state).astype('int64'))
        state = state.unsqueeze(0)
        target_list.append(state)
    target_list = torch.cat(target_list, dim=0)

    return target_list

def OA_per_class(targets, predict, classnum=11):
    result_list = np.zeros((classnum,1), dtype=np.int32)
    correct = np.zeros_like(result_list)
    total = np.zeros_like(result_list)
    for i in range(0,len(targets)):
        p_label = targets[i]
        t_label = predict[i]
        if p_label == t_label:
            correct[p_label] += 1
        total[t_label] += 1

    result = correct/total
    return result

def EVAL(pth_path, excel_path):

    MODEL_name = pth_path.split('/')[-2]
    eval_model = 2
    if eval_model == 1:
        classes_number = 11
        OA_labels = torch.load(os.path.join(pth_path, "label2.pth")) 
        model_predict = torch.load(os.path.join(pth_path, "predict2.pth"))
        
        
        onehot_label = get_onehot(OA_labels, classnum=classes_number)
        _, OA_predict = torch.max(model_predict.data, 1)
        PRC_predict = torch.softmax(model_predict, dim=1).data


        score = average_precision_score(onehot_label, PRC_predict, average='micro')
        OA = 100.* OA_predict.eq(OA_labels.data).cpu().sum().item()/OA_labels.size(0)
        OA_Per_Class = OA_per_class(OA_labels, OA_predict, classnum=classes_number)
        print("OA=%f, score=%f " % (OA, score))
        print("per_class_acc = ", end="")
        for i in OA_Per_Class:
            print(i, end="")
        
        record_list = []
        for i in OA_Per_Class:
            record_list.append(float(i))
        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([MODEL_name])
            writer.writerow(["OA=", "score= "])
            writer.writerow([OA, score])
            writer.writerow(record_list)

    
    elif eval_model == 2:
        classes_number = 11
        OA_labels = torch.load(os.path.join(pth_path, "label2.pth"))
        model_predict = torch.load(os.path.join(pth_path, "predict2.pth"))


        onehot_label = get_onehot(OA_labels, classnum=classes_number) 
        _, OA_predict = torch.max(model_predict.data, 1)
        PRC_predict = torch.softmax(model_predict, dim=1).data


        score = average_precision_score(onehot_label, PRC_predict, average='micro')
        OA = 100.* OA_predict.eq(OA_labels.data).cpu().sum().item()/OA_labels.size(0)
        OA_Per_Class = OA_per_class(OA_labels, OA_predict, classnum=classes_number)
        print("OA=%f, score=%f " % (OA, score))
        print("per_class_acc = ", end="")
        for i in OA_Per_Class:
            print(i, end="")
        print("\n", end="")

        record_list = []
        for i in OA_Per_Class:
            record_list.append(float(i))
        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([["="*20]])
            writer.writerow([MODEL_name])
            writer.writerow(["level2"])
            writer.writerow(["OA=", "score= "])
            writer.writerow([OA, score])
            writer.writerow(record_list)
        
        classes_number = 4
        OA_labels = torch.load(os.path.join(pth_path, "label1.pth")) 
        model_predict = torch.load(os.path.join(pth_path, "predict1.pth"))
        
        
        onehot_label = get_onehot(OA_labels, classnum=classes_number) 
        _, OA_predict = torch.max(model_predict.data, 1)
        PRC_predict = model_predict.data


        score = average_precision_score(onehot_label, PRC_predict, average='micro')
        OA = 100.* OA_predict.eq(OA_labels.data).cpu().sum().item()/OA_labels.size(0)
        OA_Per_Class = OA_per_class(OA_labels, OA_predict, classnum=classes_number)
        print("OA=%f, score=%f " % (OA, score))
        print("per_class_acc = ", end="")
        for i in OA_Per_Class:
            print(i, end="")

        record_list = []
        for i in OA_Per_Class:
            record_list.append(float(i))
        with open(excel_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(["level1"])
            writer.writerow(["OA=", "score= "])
            writer.writerow([OA, score])
            writer.writerow(record_list)


if __name__ == '__main__':
    
    print(0)
