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


def shift_point_cloud(batch_data, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data


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
        
def regularizer_loss(mat):
    dim = mat.shape[-1]
    ident = torch.eye(dim).to(device)
    reg_loss = torch.mean(torch.norm(torch.bmm(mat, mat.transpose(2, 1)) - ident, dim=(1, 2)))
    return reg_loss
        
def train(model, t_dataloader, v_dataloader):
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
        pclouds = torch.cat(new_clouds)
        
        points = shift_point_cloud(pclouds)
        points = torch.tensor(points)
        optimizer.zero_grad()
        output, feat_mat = model(points.to(device))
        reg_loss = regularizer_loss(feat_mat)
#         print(F.nll_loss(output, labels.to(device)), reg_loss*0.001)
        loss = F.cross_entropy(output, labels.to(device)) + reg_loss*0.001
        
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
        output, _ = model(points.to(device))
        validation_result += [i.item() for i in output.argmax(1)]
        validation_truth += [i.item() for i in labels]
    return running_loss/total_steps, training_result, training_truth, validation_result, validation_truth