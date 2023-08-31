
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F

import os
import matplotlib.pyplot as plt
import itertools
import pickle
import models

from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from PIL import Image
from torch.utils.data import Dataset, DataLoader


device = 'cuda' if torch.cuda.is_available() else 'cpu'

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix
    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']
    title:        the text to display at the top of the matrix
    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see:
                  http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues
    normalize:    If False, plot the raw numbers
                  If True, plot the proportions
    Usage
    -----
    plot_confusion_matrix(cm = cm,
                           normalize = True, # show proportions
                          target_names = y_labels_vals, # list of classes names
                          title = best_estimator_name) # title of graph
    """

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.rcParams['font.size'] = 20
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45, size=20)
        plt.yticks(tick_marks, target_names, size=20)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=20)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=20)

    plt.tight_layout()
    plt.ylabel('True label', size=20)
    plt.xlabel('Predicted label', size=20)
    # plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig('./result/MirrorNet.jpg', bbox_inches='tight')
    print('Saving Confusion Matrix..')
    plt.show()

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()  
    txt_data = [line.strip() for line in lines]  
    return txt_data

class CovidCTDataset(Dataset):
    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        """
        Args:
            txt_path (string): Path to the txt file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        File structure:
        - root_dir
            - CT_COVID
                - img1.png
                - img2.png
                - ......
            - CT_NonCOVID
                - img1.png
                - img2.png
                - ......
        """
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['COVID', 'Normal']
        self.num_cls = len(self.classes)
        self.img_list = []
        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir, self.classes[c], item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()  

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image,
                  'label': int(self.img_list[idx][1])}  
        return sample

testset = CovidCTDataset(root_dir='./data/',
                         txt_COVID='./path/COVID/path_test_covid.txt',
                         txt_NonCOVID='./path/Normal/path_test_normal.txt',
                         transform=val_transformer)

print(testset.__len__())

test_loader = DataLoader(testset, batch_size=1, drop_last=False, shuffle=False)

def test():

    model = models.Alexnet_mirror()
    model = model.to(device)

    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True

    model.load_state_dict(torch.load('./result/MirrorNet.pth'))
    model.eval()

    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []

        for batch_index, batch_samples in enumerate(test_loader):

            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output, ft = model(data)

            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, pred.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

        TN = ((predlist == 1) & (targetlist == 1)).sum()
        TP = ((predlist == 0) & (targetlist == 0)).sum()
        FP = ((predlist == 0) & (targetlist == 1)).sum()
        FN = ((predlist == 1) & (targetlist == 0)).sum()

        print('TP=', TP, 'TN=', TN, 'FN=', FN, 'FP=', FP)
        
        acc = (TP + TN) / (TP + TN + FP + FN)
        print('acc', acc)

        r = TP / (TP + FN)
        print('sensitivity', r)

        s = TN / (TN + FP)
        print('specificity', s)
        
        p = TP / (TP + FP)
        print('PPV', p)

        n = TN / (TN + FN)
        print('NPV', n)
        
        AUC = roc_auc_score(targetlist, scorelist)
        print('AUC', AUC)

        C_M = confusion_matrix(targetlist, predlist)
        print('confusion matrix', C_M)
        plot_confusion_matrix(C_M, normalize=False, target_names=['COVID-19', 'Normal'], title='Confusion matrix')

        fpr, tpr, _ = roc_curve(targetlist, scorelist)
        with open('./result/MirrorNet_roc.pkl', "wb") as fw:
            pickle.dump((fpr, tpr, AUC), fw)
        with open('./result/MirrorNet_roc.pkl', "rb") as fr:
            FPR, TPR, AUC = pickle.load(fr)

        plt.figure(figsize=(8, 6))
        plt.plot(FPR, TPR, color='red', linewidth=2, label='MirrorNet  AUC = %.4f' % AUC)
        plt.xlabel('False Positive Rate', size=20)
        plt.ylabel('True Positive Rate', size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.title('ROC curve', fontsize=20)
        plt.legend(loc="lower right", prop={'size': 16})
        plt.savefig('./result/ROC_curve.png')

test()
