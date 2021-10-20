import torch
import numpy as np

for shots in ['1shots','5shots']:
    for i in range (65):
        class_accuracy =  torch.load('exp_proj/chunks/class_accuracy_'+ shots +str(i)).unsqueeze(0)
        if i==0:
            proj_class_acc=class_accuracy
        else:
            proj_class_acc=torch.cat((proj_class_acc,class_accuracy))


    torch.save(proj_class_acc,'exp_proj/complete_class_accuracy'+shots)