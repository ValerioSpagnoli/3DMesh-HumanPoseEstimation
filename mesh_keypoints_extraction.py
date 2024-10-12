from extract_edge_features import extract_edge_features

import os
import pandas as pd
import numpy as np
import trimesh
import plotly.graph_objects as go
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

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
    
    self.fc_phi = nn.Linear(256, 128) 
    self.fc_rho = nn.Linear(128, num_keypoints * 3)
      
  def forward(self, x):
    x = x.permute(0, 1, 3, 2)
    
    x = x.permute(0, 3, 1, 2)
    x = torch.relu(self.conv1(x))
    x = torch.relu(self.conv2(x))
    x = torch.relu(self.conv3(x))
    
    x = self.pool(x)
    x = x.view(x.size(0), -1)
    
    x = torch.relu(self.fc_phi(x))
    x = self.fc_rho(x)
    
    keypoints = x.view(x.size(0), -1, 3)
    return keypoints


class ChamferLoss(nn.Module):
  def __init__(self, device='cpu'):
    super(ChamferLoss, self).__init__()
    self.device = device

  def forward(self, predicted_points, target_points):
    distance_predicted_to_target = torch.cdist(predicted_points, target_points)
    min_distance_predicted_to_target, _ = torch.min(distance_predicted_to_target, dim=2)
    min_distance_target_to_predicted, _ = torch.min(distance_predicted_to_target, dim=1)
    loss = torch.mean(min_distance_predicted_to_target) + torch.mean(min_distance_target_to_predicted)
    return loss.to(self.device)
  
  
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
    
    if (epoch+1) % 10 == 0:
      torch.save(keypoints_predictor.state_dict(), model_save_dir + f'checkpoints/keypoints_predictor_{epoch+1}.pth')

  torch.save(keypoints_predictor.state_dict(), model_save_dir + 'keypoints_predictor.pth')
  

def test(keypoints_predictor, test_loader, criterion, device):
  keypoints_predictor.eval()
  test_loss = 0.0
  with torch.no_grad():
    for _, inputs, labels in tqdm(test_loader, desc=f"Testing"):
        inputs, labels = inputs.to(torch.float32).to(device), labels.to(torch.float32).to(device)
        with autocast(device_type='cuda'):
          outputs = keypoints_predictor(inputs)
          loss = criterion(outputs, labels)
        test_loss += loss.item()
  test_loss = test_loss/len(test_loader)
  print(f'Test Loss: {test_loss}')


def test_single_mesh(keypoints_predictor, mesh, edge_features, keypoints=None, device='cpu'):
  keypoints = keypoints.cpu().numpy()
  predicted_keypoints = keypoints_predictor(edge_features.unsqueeze(0).to(torch.float32).to(device)).squeeze().cpu().detach().numpy()

  fig = go.Figure()
  fig.update_layout(scene=dict(aspectmode='data'))

  fig.add_trace(go.Mesh3d(x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color='lightgrey', opacity=0.5))

  for i, keypoint in enumerate(predicted_keypoints):
      fig.add_trace(go.Scatter3d(x=[keypoint[0]], y=[keypoint[1]], z=[keypoint[2]], mode='markers', marker=dict(size=5, color='blue')))
  
  if keypoints is not None:
    for i, keypoint in enumerate(keypoints):
        fig.add_trace(go.Scatter3d(x=[keypoint[0]], y=[keypoint[1]], z=[keypoint[2]], mode='markers', marker=dict(size=2, color='red'))) 
        
  fig.show()
  

if __name__ == '__main__':
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

  dataset_dir = 'datasets/mesh_keypoints_extraction'
  meshes_dir = os.path.join(dataset_dir, 'meshes')
  keypoints_dir = os.path.join(dataset_dir, 'keypoints')
  model_save_dir = 'models/'

  num_edges = 750
  input_channels = 5
  num_keypoints = 12

  batch_size = 32
  learning_rate = 0.001
  num_epochs = 150
  
  dataset = MeshData(meshes_dir, keypoints_dir, device=device, num_edges=num_edges, normalize=True)
  train_set_size = int(0.8 * len(dataset))
  val_set_size = int(0.1 * len(dataset))
  test_set_size = len(dataset) - train_set_size - val_set_size
  train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_set_size, val_set_size, test_set_size])

  train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
  valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
  test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
  
  keypoints_predictor = KeypointPredictionNetwork(input_channels=input_channels, num_keypoints=num_keypoints).to(device)
  optimizer = optim.Adam(keypoints_predictor.parameters(), lr=learning_rate)
  criterion = ChamferLoss(device=device)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=3)
  scaler = GradScaler()

  train(keypoints_predictor, optimizer, criterion, scaler, scheduler, train_loader, valid_loader, num_epochs, device, model_save_dir)
  
  keypoints_predictor_test = KeypointPredictionNetwork(input_channels=input_channels, num_keypoints=num_keypoints).to(device)
  keypoints_predictor_test.load_state_dict(torch.load(model_save_dir + 'keypoints_predictor.pth'))
  test(keypoints_predictor_test, test_loader, device)