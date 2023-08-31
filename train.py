
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

import os
import warnings
import models

from sklearn.metrics import roc_auc_score, confusion_matrix
from PIL import Image
from torch.utils.data import Dataset, DataLoader


warnings.filterwarnings('ignore')
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_accuracy = 0
batchsize = 1

print('==> Preparing data..')


train_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


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


trainset = CovidCTDataset(root_dir='./data',
                          txt_COVID='./path/COVID/path_train_covid.txt',
                          txt_NonCOVID='./path/Normal/path_train_normal.txt',
                          transform=train_transformer)

valset = CovidCTDataset(root_dir='./data/',
                         txt_COVID='./path/COVID/path_test_covid.txt',
                         txt_NonCOVID='./path/Normal/path_test_normal.txt',
                         transform=val_transformer)

print(trainset.__len__())
print(valset.__len__())

train_loader = DataLoader(trainset, batch_size=batchsize, drop_last=False, shuffle=True)
val_loader = DataLoader(valset, batch_size=batchsize, drop_last=False, shuffle=False)

print('==> Building model..')

model = models.Alexnet_mirror()
modelname = 'MirrorNet'
model = model.to(device)


if device == 'cuda':
    model = torch.nn.DataParallel(model)
    cudnn.benchmark = True

optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
# scheduler.step()

criterion1 = nn.CrossEntropyLoss()
criterion2 = nn.MSELoss()

def train(epoch):

    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    train_correct = 0

    for batch_index, batch_samples in enumerate(train_loader):

        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

        optimizer.zero_grad()
        output, ft = model(data)
        ftf = torch.flip(ft, [3])
        alp = 1
        if target[0] == 1:
           alp = 0

        loss1 = criterion1(output, target.long())
        loss2 = criterion2(ft, ftf)
        loss = loss1 + loss2 * 0.1 * alp
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        predicted = output.argmax(dim=1, keepdim=True)
        train_correct += predicted.eq(target.long().view_as(predicted)).sum().item()

    print('Train set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))

    f = open('result/{}_train.txt'.format(modelname), 'a+')
    f.write('\nTrain set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        train_loss / len(train_loader.dataset), train_correct, len(train_loader.dataset),
        100.0 * train_correct / len(train_loader.dataset)))
    f.close()


def val(epoch):

    global best_accuracy
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():

        predlist = []
        scorelist = []
        targetlist = []

        for batch_index, batch_samples in enumerate(val_loader):

            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            output, ft = model(data)
            ftf = torch.flip(ft, [3])
            alp = 1
            if target[0] == 1:
                alp = 0

            loss1 = criterion1(output, target.long())
            loss2 = criterion2(ft, ftf)
            loss = loss1 + loss2 * 0.1 * alp
            test_loss += loss.item()

            score = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1, keepdim=True)
            correct += predicted.eq(target.long().view_as(predicted)).sum().item()

            targetcpu = target.long().cpu().numpy()
            predlist = np.append(predlist, predicted.cpu().numpy())
            scorelist = np.append(scorelist, score.cpu().numpy()[:, 1])
            targetlist = np.append(targetlist, targetcpu)

        print('Val set: loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss/len(val_loader.dataset), correct, len(val_loader.dataset),
            100.0 * correct / len(val_loader.dataset)))

        TN = ((predlist == 1) & (targetlist == 1)).sum()
        TP = ((predlist == 0) & (targetlist == 0)).sum()
        FP = ((predlist == 0) & (targetlist == 1)).sum()
        FN = ((predlist == 1) & (targetlist == 0)).sum()

        print('TP =', TP, 'TN =', TN, 'FN =', FN, 'FP =', FP)

        acc = (TP + TN) / (TP + TN + FP + FN)
        r = TP / (TP + FN)       
        s = TN / (TN + FP)
        p = TP / (TP + FP)
        n = TN / (TN + FN)
        AUC = roc_auc_score(targetlist, scorelist)
        C_M = confusion_matrix(targetlist, predlist)

        print('accuracy: {:.4f}, sensitivity: {:.4f}, specificity: {:.4f}, PPV: {:.4f}, NPV: {:.4f}, AUC: {:.4f}'.format(acc, r, s, p, n, AUC))

        f = open('result/{}_val.txt'.format(modelname), 'a+')
        f.write('\naccuracy: {:.4f}, sensitivity: {:.4f}, specificity: {:.4f}, PPV: {:.4f}, NPV: {:.4f}, AUC: {:.4f}'.format(acc, r, s, p, n, AUC))
        f.close()

        if acc > best_accuracy:
            torch.save(model.state_dict(), './result/{}_model.pth'.format(modelname))
            print("Saving FPR, TPR and model")
            best_accuracy = acc


for epoch in range(0, 300):
    train(epoch)
    val(epoch)
    scheduler.step()
