import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from random import sample

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Cuda available: ",torch.cuda.is_available())
print("Current device: ",  torch.cuda.current_device())


class TNet(nn.Module):
    def __init__(self, input_dim):
        super(TNet, self).__init__()
        self.input_dim  = input_dim
        self.mlp1 = nn.Linear(self.input_dim, 64)
        
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)
        
        self.mlp4 = nn.Linear(256, 512)
        self.mlp5 = nn.Linear(512, 256)
        self.mlp6 = nn.Linear(256, self.input_dim**2)
        
        self.bn1 = nn.BatchNorm1d(64, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(256, track_running_stats=False)
        
        self.bn4 = nn.BatchNorm1d(512, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(256, track_running_stats=False)
        self.softmax = torch.nn.Softmax(dim=1)
        self.to(device)
    
    def forward(self, pclouds):
        new_cloud = []
        for i in range(pclouds.shape[0]):
            x = F.relu(self.bn1(self.mlp1(pclouds[i])))
            x = F.relu(self.bn2(self.mlp2(x)))
            x = F.relu(self.bn3(self.mlp3(x)))
            new_cloud.append(x.unsqueeze(0))
        x = torch.cat(new_cloud)
        x = torch.max(x, 1)[0]
        x = x.view(-1, 256)
        x = F.relu(self.bn4(self.mlp4(x)))
        x = F.relu(self.bn5(self.mlp5(x)))
        x = self.mlp6(x)
        x = x.view(-1, self.input_dim, self.input_dim)
        ident = torch.tensor(np.diag(np.ones(self.input_dim))).float().repeat(x.shape[0], 1, 1).to(device)
        x = torch.add(x, ident)
        return x
    
class PointNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PointNet, self).__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim
        self.tnet1 = TNet(3)
        self.tnet2 = TNet(128)
        
        self.mlp1 = nn.Linear(self.input_dim, 64)
        self.mlp2 = nn.Linear(64, 128)
        self.mlp3 = nn.Linear(128, 256)
        self.mlp4 = nn.Linear(256, 512)
        self.mlp5 = nn.Linear(512, 256)
        self.mlp6 = nn.Linear(256, self.output_dim)

        self.bn1 = nn.BatchNorm1d(64, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(128, track_running_stats=False)
        self.bn3 = nn.BatchNorm1d(256, track_running_stats=False)
        
        self.bn4 = nn.BatchNorm1d(512, track_running_stats=False)
        self.bn5 = nn.BatchNorm1d(256, track_running_stats=False)
#         self.softmax = torch.nn.Softmax(dim=1)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.to(device)
    
    def forward(self, pclouds):
#         pos = (pclouds - torch.mean(pclouds, 1))/torch.std(pclouds, 1)
        dim = pclouds.shape
        feat1 = self.tnet1(pclouds)
        out = torch.bmm(pclouds, feat1.transpose(2, 1)).squeeze(2)
        
        new_clouds = []
        for i in range(out.shape[0]):
            x = F.relu(self.bn1(self.mlp1(out[i])))
            x = F.relu(self.bn2(self.mlp2(x)))
            new_clouds.append(x.unsqueeze(0))
        pclouds = torch.cat(new_clouds)
        feat2 = self.tnet2(pclouds)
        out = torch.bmm(pclouds, feat2.transpose(2, 1)).squeeze(2)
        
        new_clouds = []
        for i in range(out.shape[0]):
            x = F.relu(self.bn3(self.mlp3(out[i])))
            new_clouds.append(x.unsqueeze(0))
        pclouds = torch.cat(new_clouds)
        x = torch.max(pclouds, 1)[0]
        x = x.view(-1, 256)
        x = self.dropout1(x)
        x = F.relu(self.bn4(self.mlp4(x)))
        x = self.dropout2(x)
        x = F.relu(self.bn5(self.mlp5(x)))
        return F.log_softmax(self.mlp6(x)), feat2

class MAB(nn.Module):
    def __init__(self, embed_dim):
        super(MAB, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 8)
        self.rff = nn.Linear(embed_dim, embed_dim)
        self.to(device)
        
    def forward(self, Y, X):
        attn_output, _ = self.multihead_attn(Y, X, X)
        H = attn_output + Y
        H_output = self.rff(H)
        output = H_output + H
        return output
    
class SAB(nn.Module):
    def __init__(self, embed_dim):
        super(SAB, self).__init__()
        self.mab = MAB(embed_dim)
        self.to(device)
        
    def forward(self, X):
        output = self.mab(X, X)
        return output
    
class PoolMA(nn.Module):
    def __init__(self, embed_dim):
        super(PoolMA, self).__init__()
        self.mab = MAB(embed_dim)
        self.rff = nn.Linear(embed_dim, embed_dim)
        self.s_param = nn.Parameter(torch.Tensor(1, 1, embed_dim))
        nn.init.xavier_uniform_(self.s_param)
        self.to(device)
    
    def forward(self, Z):
        z = self.rff(Z)
        attn_output = self.mab(self.s_param.repeat(1, Z.shape[1], 1), z)
        return attn_output
    
class SetTransformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim):
        super(Set_Transformer, self).__init__()
        self.lin1 = nn.Linear(input_dim, embed_dim)
        self.sab1 = SAB(embed_dim)
        self.sab2 = SAB(embed_dim)
        self.sab3 = SAB(embed_dim)
        self.pool = PoolMA(embed_dim)
        self.sab4 = SAB(embed_dim)
        self.rff = nn.Linear(embed_dim, output_dim)
        self.to(device)
    
    def forward(self, Z):
        x = self.lin1(Z)
        x = self.sab1(x)
        x = self.sab2(x)
        x = self.sab3(x)
        x = self.pool(x)
        x = self.sab4(x)
        x = self.rff(x)
        return x
    
def regularizer_loss(mat):
    dim = mat.shape[-1]
    ident = torch.eye(dim).to(device)
    reg_loss = torch.mean(torch.norm(torch.bmm(mat, mat.transpose(2, 1)) - ident, dim=(1, 2)))
    return reg_loss


def train_set_transformer(model, t_dataloader, v_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    running_loss = 0
    total_steps = 0
    training_result, training_truth = [], []
    validation_result, validation_truth = [], []
    model.train()
    for data in t_dataloader:
        labels = data.y
        cloud_list = data.pos.split(1000)
        new_clouds = []
        for j in range(len(cloud_list)):
            new_points = cloud_list[j]
            new_points = (new_points - torch.mean(new_points))/torch.std(new_points)
            new_clouds.append(new_points.unsqueeze(0))
        points = torch.cat(new_clouds)
        
#         points = shift_point_cloud(pclouds)
        points = torch.tensor(points).permute([1, 0, 2])
        optimizer.zero_grad()
        # Pointnet output
#         output, feat_mat = model(points.to(device))
#         reg_loss = regularizer_loss(feat_mat)
#         print(F.nll_loss(output, labels.to(device)), reg_loss*0.001)
#         loss = F.cross_entropy(output, labels.to(device)) + reg_loss*0.001

        # Set Transformer output
        output = model(points.to(device)).squeeze(0)
        loss = F.cross_entropy(output, labels.to(device)) 
        training_result += [i.item() for i in output.argmax(1)]
        training_truth += [i.item() for i in labels]
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
        total_steps += 1
    model.eval()
    for data in v_dataloader:
        labels = data.y
        cloud_list = data.pos.split(1000)
        new_clouds = []
        for j in range(len(cloud_list)):
            new_points = cloud_list[j]
            new_points = (new_points - torch.mean(new_points))/torch.std(new_points)
            new_clouds.append(new_points.unsqueeze(0))
        pclouds = torch.cat(new_clouds)
        points = torch.tensor(pclouds).permute([1, 0, 2])
        # Pointnet output
#         output, _ = model(points.to(device))
        # Set Transformer output
        output = model(points.to(device)).squeeze(0)
        validation_result += [i.item() for i in output.argmax(1)]
        validation_truth += [i.item() for i in labels]
    return running_loss/total_steps, training_result, training_truth, validation_result, validation_truth


def train_pointnet(model, t_dataloader, v_dataloader):
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    running_loss = 0
    total_steps = 0
    training_result, training_truth = [], []
    validation_result, validation_truth = [], []
    model.train()
    for data in t_dataloader:
        labels = data.y
        cloud_list = data.pos.split(1000)
        new_clouds = []
        for j in range(len(cloud_list)):
            new_points = cloud_list[j]
            new_points = (new_points - torch.mean(new_points))/torch.std(new_points)
            new_clouds.append(new_points.unsqueeze(0))
        points = torch.cat(new_clouds)
       
        points = torch.tensor(points)
        optimizer.zero_grad()
        # Pointnet output
        output, feat_mat = model(points.to(device))
        reg_loss = regularizer_loss(feat_mat)
        loss = F.cross_entropy(output, labels.to(device)) + reg_loss*0.001
        loss = F.cross_entropy(output, labels.to(device)) 
        training_result += [i.item() for i in output.argmax(1)]
        training_truth += [i.item() for i in labels]
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
        total_steps += 1
    model.eval()
    for data in v_dataloader:
        labels = data.y
        cloud_list = data.pos.split(1000)
        new_clouds = []
        for j in range(len(cloud_list)):
            new_points = cloud_list[j]
            new_points = (new_points - torch.mean(new_points))/torch.std(new_points)
            new_clouds.append(new_points.unsqueeze(0))
        pclouds = torch.cat(new_clouds)
        points = torch.tensor(pclouds)
        # Pointnet output
        output, _ = model(points.to(device))
        validation_result += [i.item() for i in output.argmax(1)]
        validation_truth += [i.item() for i in labels]
    return running_loss/total_steps, training_result, training_truth, validation_result, validation_truth