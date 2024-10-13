from extract_edge_features import extract_edge_features

import os
import pandas as pd
import numpy as np
import trimesh
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.amp import autocast
from scipy.optimize import linear_sum_assignment

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

class MeshData(Dataset):
  def __init__(self, mesh_dir, keypoints_dir, device, num_edges=1024, normalize=True):
    self.mesh_dir = mesh_dir
    self.keypoints_dir = keypoints_dir
    self.device = device
    self.num_edges = num_edges
    self.normalize = normalize  
    
    self.mesh_files = sorted(os.listdir(mesh_dir))
    self.keypoints_files = sorted(os.listdir(keypoints_dir))
    
  def __len__(self):
    return len(self.mesh_files)
  
  def __getitem__(self, idx):
    #* Get mesh
    mesh_path = os.path.join(self.mesh_dir, self.mesh_files[idx]) 
    mesh_obj = trimesh.load(mesh_path)
        
    #* Get keypoints
    keypoints = pd.read_csv(os.path.join(self.keypoints_dir, self.keypoints_files[idx]))[['x', 'y', 'z']].values
    
    #* Normalize mesh and keypoints
    if self.normalize:    
      centroid = mesh_obj.centroid
      scale = mesh_obj.scale
      mesh_obj.vertices -= centroid
      mesh_obj.vertices /= scale
      keypoints[:, :3] -= centroid
      keypoints[:, :3] /= scale
        
    #* Get edge features
    edge_features = extract_edge_features(mesh_obj, device=self.device)
    if edge_features.shape[0] > self.num_edges:
      edge_features = edge_features[:self.num_edges, :]
    else:
      padding = torch.zeros(self.num_edges - edge_features.shape[0], edge_features.shape[1]).to(self.device)  
      edge_features = torch.cat([edge_features, padding], dim=0)
    
    
    edge_features = edge_features.to(self.device) 
    keypoints = torch.tensor(keypoints).float().to(self.device)
    return mesh_obj, edge_features, keypoints
  
  
def custom_collate_fn(batch):
  meshes = []
  edge_features_list = []
  keypoints_list = []

  for mesh_obj, edge_features, keypoints in batch:
    meshes.append(mesh_obj)
    edge_features_list.append(edge_features)
    keypoints_list.append(keypoints)

  edge_features_batch = torch.stack(edge_features_list)
  keypoints_batch = torch.stack(keypoints_list)

  return meshes, edge_features_batch, keypoints_batch
  
  
class KeypointPredictionNetwork(nn.Module):
  def __init__(self, input_channels=5, num_keypoints=12):
    super(KeypointPredictionNetwork, self).__init__()
    
    self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

    self.pool = nn.AdaptiveMaxPool2d((1, 1))
    
    self.fc1 = nn.Linear(256, 128) 
    self.fc2 = nn.Linear(128, num_keypoints * 3)
      
  def forward(self, x):
    x = x.permute(0, 1, 3, 2)
    
    x = x.permute(0, 3, 1, 2)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    
    x = torch.relu(self.fc1(x))
    x = self.fc2(x)
    
    keypoints = x.view(x.size(0), -1, 3)
    return keypoints


class HungarianSumOfDistancesLoss(nn.Module):
    def __init__(self):
        super(HungarianSumOfDistancesLoss, self).__init__()

    def forward(self, pred, target):
        B, N, _ = pred.size()
        total_distance = 0.0
        for b in range(B):
            distance_matrix = torch.cdist(pred[b], target[b], p=2)  # (N, N)
            distance_matrix_np = distance_matrix.cpu().detach().numpy()
            row_indices, col_indices = linear_sum_assignment(distance_matrix_np)
            matched_distances = distance_matrix[row_indices, col_indices].sum()
            total_distance += matched_distances

        loss = total_distance / B
        return loss


def train(keypoints_predictor, optimizer, criterion, scaler, scheduler, train_loader, valid_loader, num_epochs, device, model_save_dir):
  for epoch in range(num_epochs):
    keypoints_predictor.train()
    train_loss = 0.0
    print(f"Epoch {epoch+1}/{num_epochs}")
    for _, inputs, labels in tqdm(train_loader, desc=f" - Training"):
      inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
      
      optimizer.zero_grad()
      
      with autocast(device_type='cuda'):    
        outputs = keypoints_predictor(inputs)
        loss = criterion(outputs, labels)

      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      
      train_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    scheduler.step(train_loss)

    keypoints_predictor.eval()
    valid_loss = 0.0
    with torch.no_grad():
      for _, inputs, labels in tqdm(valid_loader, desc=f" - Validation"):
        inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
        
        with autocast(device_type='cuda'):    
          outputs = keypoints_predictor(inputs)
          loss = criterion(outputs, labels)

        valid_loss += loss.item()
        
    valid_loss = valid_loss/len(valid_loader)
    
    print(f' - Train Loss: {train_loss} - Valid Loss: {valid_loss} - Learning Rate: {scheduler.get_last_lr()[0]}')
    
    # if (epoch+1) % 10 == 0:
    torch.save(keypoints_predictor.state_dict(), model_save_dir + f'checkpoints/keypoints_predictor_{epoch+1}.pth')

  torch.save(keypoints_predictor.state_dict(), model_save_dir + 'keypoints_predictor.pth')


def hungarian_mpjpe(pred, target):
    B, N, _ = pred.size()

    total_error = 0.0

    for b in range(B):
        distance_matrix = torch.cdist(pred[b].to(torch.float32), target[b].to(torch.float32), p=2)
        distance_matrix_np = distance_matrix.cpu().detach().numpy()
        row_indices, col_indices = linear_sum_assignment(distance_matrix_np)
        matched_distances = distance_matrix[row_indices, col_indices]
        total_error += matched_distances.sum()

    mpjpe = total_error / (B * N)
    return mpjpe

def hungarian_pck(pred, target, threshold=0.1):
    B, N, _ = pred.size()

    correct_keypoints = 0.0
    total_keypoints = B * N

    for b in range(B):
        pred_b = pred[b].float() if pred.dtype == torch.float16 else pred[b]
        target_b = target[b].float() if target.dtype == torch.float16 else target[b]
        distance_matrix = torch.cdist(pred_b, target_b, p=2)
        distance_matrix_np = distance_matrix.cpu().detach().numpy()
        row_indices, col_indices = linear_sum_assignment(distance_matrix_np)
        matched_distances = distance_matrix[row_indices, col_indices]
        correct_keypoints += (matched_distances <= threshold).float().sum().item()
        
    pck_score = correct_keypoints / total_keypoints
    return pck_score


def test(keypoints_predictor, test_loader, criterion, device):
  keypoints_predictor.eval()
  
  test_loss = 0.0
  mpjpe = 0.0
  pck = 0.0
  
  with torch.no_grad():
    for _, inputs, labels in tqdm(test_loader, desc=f"Testing"):
        inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
        
        with autocast(device_type='cuda'):
          outputs = keypoints_predictor(inputs)
          loss = criterion(outputs, labels)
        
        test_loss += loss.item()
        mpjpe += hungarian_mpjpe(outputs, labels)
        pck += hungarian_pck(outputs, labels)
  
  test_loss = test_loss/len(test_loader)
  mpjpe = mpjpe/len(test_loader)
  pck = pck/len(test_loader)
  
  test_loss = round(float(test_loss), 4)
  mpjpe = round(float(mpjpe), 4)
  pck = round(float(pck), 4)
  
  print(f'Test results:')
  print(f' - Test Loss: {test_loss}')
  print(f' - MPJPE:     {mpjpe} [m]')
  print(f' - PCK:       {pck*100} %')
  

def test_single_mesh(keypoints_predictor, edge_features, keypoints, criterion, device):
  edge_features = edge_features.unsqueeze(0).to(torch.float32).to(device)
  predicted_keypoints = keypoints_predictor(edge_features)

  loss = criterion(predicted_keypoints, keypoints.unsqueeze(0).to(torch.float32).to(device))
  mpjpe = hungarian_mpjpe(predicted_keypoints, keypoints.unsqueeze(0).to(torch.float32).to(device))
  pck = hungarian_pck(predicted_keypoints, keypoints.unsqueeze(0).to(torch.float32).to(device))
  
  loss = round(float(loss), 4)
  mpjpe = round(float(mpjpe), 4)
  pck = round(float(pck), 4)
  
  print(f'Test results:')
  print(f' - Test Loss: {loss}')
  print(f' - MPJPE:     {mpjpe} [m]')
  print(f' - PCK:       {pck*100} %')
  
  return predicted_keypoints.cpu().detach().numpy()[0]