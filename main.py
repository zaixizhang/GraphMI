import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from models.gcn import GCN, embedding_GCN
from topology_attack import PGDAttack
from utils import *
from dataset import Dataset
import argparse
from sklearn.metrics import roc_curve, auc, average_precision_score
import scipy.io as sio
import random
import os

def test1(adj, features, labels, victim_model):
    adj, features, labels = to_tensor(adj, features, labels, device=device)

    victim_model.eval()
    adj_norm = normalize_adj_tensor(adj)
    output = victim_model(features, adj_norm)

    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    return output.detach()

def test(adj):
    ''' test on GCN '''

    # adj = normalize_adj_tensor(adj)
    gcn = GCN(nfeat=features.shape[1],
              nhid=args.hidden,
              nclass=labels.max().item() + 1,
              dropout=args.dropout, device=device)
    gcn = gcn.to(device)
    gcn.fit(features, adj, labels, idx_train)  # train without model picking
    # gcn.fit(features, adj, labels, idx_train, idx_val) # train with validation model picking
    output = gcn.output.cpu()
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    return acc_test.item()

def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]))
    return adj

def preprocess_Adj(adj, feature_adj):
    n=len(adj)
    cnt=0
    adj=adj.numpy()
    feature_adj=feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j]>0.14 and adj[i][j]==0.0:
                adj[i][j]=1.0
                cnt+=1
    print(cnt)
    return torch.FloatTensor(adj)

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def AP(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    count = 0
    for i in idx:
        for j in idx:
            if i != j and ori_adj[i][j] == 1 and count <= 500:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])
                count += 1
    count = 0
    #negtive edge sampling
    x = np.random.choice(idx, 500)
    y = np.random.choice(idx, 500)
    for i in range(500):
        if x[i] != y[i] and ori_adj[i][j] == 0:
            real_edge.append(0)
            pred_edge.append(modified_adj[i][j])
            count += 1
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    print(average_precision_score(real_edge, pred_edge))

def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])
                #pred_edge.append(np.dot(output[idx[i]], output[idx[j]])/(np.linalg.norm(output[idx[i]])*np.linalg.norm(output[idx[j]])))
                #pred_edge.append(-np.linalg.norm(output[idx[i]]-output[idx[j]]))
                #pred_edge.append(np.dot(features[idx[i]], features[idx[j]]) / (np.linalg.norm(features[idx[i]]) * np.linalg.norm(features[idx[j]])))

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)

def result(ori_adj, idx, mask):
    t = 0
    tp = 0
    pt = 0
    for i in idx:
        for j in idx:
            if i != j and ori_adj[i][j] == 1:
                t += 1
                if mask[i][j] ==1:
                    tp += 1
            if i !=j and mask[i][j] == 1:
                pt += 1
    print('recall:', tp/t, 'precision: ', tp/pt)
    return tp/t

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=1.0, help='pertubation rate')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=1.0)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

data = Dataset(root='', name=args.dataset, setting='GCN')
adj, features, labels, random_adj = data.adj, data.features, data.labels, data.random_adj

idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
idx_inversion = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))
perturbations = int(args.ptb_rate * (adj.sum() // 2))

adj, features, labels = preprocess(adj, features, labels, preprocess_adj=False, onehot_feature=False)
feature_adj = dot_product_decode(features)
#preprocess_adj = preprocess_Adj(adj, feature_adj)
random_adj = torch.FloatTensor(random_adj.todense())
# Setup Victim Model

victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
                   dropout=0.5, weight_decay=5e-4, device=device)

victim_model = victim_model.to(device)
victim_model.fit(features, adj, labels, idx_train, idx_val)

embedding = embedding_GCN(nfeat=features.shape[1], nhid=16, device=device)
embedding.load_state_dict(transfer_state_dict(victim_model.state_dict(), embedding.state_dict()))


#try pseudo label
'''
output = test1(random_adj, features, labels, victim_model)
labels = torch.argmax(output, dim=1)
labels = labels.cpu()
'''
# Setup Attack Model

model = PGDAttack(model=victim_model, embedding=embedding, nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)

def main():

    model.attack(features, random_adj, labels, idx_inversion, perturbations, epochs=args.epochs)
    modified_adj = model.modified_adj.cpu()
    #mask = model.edge_select.cpu()
    print('=== testing GCN on original(clean) graph ===')
    test1(adj, features, labels, victim_model)

    print('=== calculating link inference AUC ===')
    #Auc(adj.numpy(), modified_adj.numpy(), idx_inversion)
    #Auc(adj.numpy(), modified_adj.numpy(), idx_inversion)
    AP(adj.numpy(), modified_adj.numpy(), idx_inversion)

    output = embedding_GCN(features.to(device), torch.zeros(adj.shape[0], adj.shape[0]).to(device))
    adj1 = dot_product_decode(output.cpu())
    AP(adj.numpy(), adj1.detach().numpy(), idx_inversion)

    # # if you want to save the modified adj/features, uncomment the code below
    # model.save_adj(root='./', name=f'mod_adj')
    # model.save_features(root='./', name='mod_features')

if __name__ == '__main__':
    main()
