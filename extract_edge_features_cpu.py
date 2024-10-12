import os
import trimesh
import numpy as np
from tqdm import tqdm as tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import confusion_matrix

def meshcnn_like_features(edge_properties):
  edge_features = {
    'dihaedral_angle': 0,
    'inner_angle_1': 0,
    'inner_angle_2': 0,
    'length_height_ratio_1': 0,
    'length_height_ratio_2': 0
  }
  
  #* Compute the dihedral angle
  face_0_vertices = np.array(edge_properties['face_0']['vertices'])
  face_1_vertices = np.array(edge_properties['face_1']['vertices'])

  normal_0 = np.cross(face_0_vertices[1] - face_0_vertices[0], face_0_vertices[2] - face_0_vertices[0]).astype(np.float32)
  normal_0 = normal_0 / np.linalg.norm(normal_0)
  normal_1 = np.cross(face_1_vertices[1] - face_1_vertices[0], face_1_vertices[2] - face_1_vertices[0]).astype(np.float32)
  normal_1 = normal_1 / np.linalg.norm(normal_1)

  cos_angle = np.dot(normal_0, normal_1)
  cos_angle = np.clip(cos_angle, -1, 1)
  dihedral_angle = np.arccos(cos_angle)

  # FEATURE 1: dihaedral angle
  edge_features['dihaedral_angle'] = dihedral_angle
  
  #* Compute the inner angles
  for i in range(2):
    edge_0 = edge_properties[f'face_{i}']['neighbor_0']['vertices']
    edge_1 = edge_properties[f'face_{i}']['neighbor_1']['vertices']
    common_vertex = [v for v in edge_properties[f'face_{i}']['vertices'] if v in edge_0 and v in edge_1][0]
    vertex_1 = [v for v in edge_0 if not np.array_equal(v, common_vertex)][0]
    vertex_2 = [v for v in edge_1 if not np.array_equal(v, common_vertex)][0]
    d1 = common_vertex - vertex_1
    d1 = d1 / np.linalg.norm(d1)
    d2 = vertex_2 - common_vertex
    d2 = d2 / np.linalg.norm(d2)
    cos_angle = np.clip(np.dot(d1, d2), -1, 1)
    inner_angle = np.arccos(cos_angle)
    # FEATURE 2/3: inner angles
    edge_features[f'inner_angle_{i}'] = inner_angle
  
  #* Compute the length-height ratio  
  edge_length = np.linalg.norm(edge_properties['vertices'][0] - edge_properties['vertices'][1])
  for i in range(2):
    face_vertex_0 = edge_properties[f'face_{i}']['vertices'][0]
    face_vertex_1 = edge_properties[f'face_{i}']['vertices'][1]
    face_vertex_2 = edge_properties[f'face_{i}']['vertices'][2]
    face_area = np.linalg.norm(np.cross(face_vertex_1 - face_vertex_0, face_vertex_2 - face_vertex_0))
    length_height_ratio = (edge_length*edge_length) / (2*face_area)
    # FEATURE 4/5: length height ratio
    edge_features[f'length_height_ratio_{i}'] = length_height_ratio

  return edge_features


def simple_features(edge_properties):
  vertex_0 = edge_properties['vertices'][0]
  vertex_1 = edge_properties['vertices'][1]
  edge_length = np.linalg.norm(edge_properties['vertices'][0] - edge_properties['vertices'][1])  
  
  return np.array([vertex_0[0], vertex_0[1], vertex_0[2], vertex_1[0], vertex_1[1], vertex_1[2], edge_length])

def simple_features_2d(edge_properties):
  edge_vertex_0 = edge_properties['vertices'][0]
  edge_vertex_1 = edge_properties['vertices'][1]
  edge_length = np.linalg.norm(edge_properties['vertices'][0] - edge_properties['vertices'][1])
  edge_features = [edge_vertex_0[0], edge_vertex_0[1], edge_vertex_0[2], edge_vertex_1[0], edge_vertex_1[1], edge_vertex_1[2], edge_length]
  
  f0_n0_features = np.zeros(7)
  f0_n1_features = np.zeros(7)
  f1_n0_features = np.zeros(7)
  f1_n1_features = np.zeros(7)
  for i in range(2):
    neighbor_0 = edge_properties[f'face_{i}']['neighbor_0']['vertices']
    neighbor_0_vertex_0 = neighbor_0[0]
    neighbor_0_vertex_1 = neighbor_0[1]
    neighbor_0_length = np.linalg.norm(neighbor_0_vertex_0 - neighbor_0_vertex_1)
    neighbor_0_features = [neighbor_0_vertex_0[0], neighbor_0_vertex_0[1], neighbor_0_vertex_0[2], neighbor_0_vertex_1[0], neighbor_0_vertex_1[1], neighbor_0_vertex_1[2], neighbor_0_length]
    if i == 0: f0_n0_features = neighbor_0_features
    elif i == 1: f1_n0_features = neighbor_0_features
    
    neighbor_1 = edge_properties[f'face_{i}']['neighbor_1']['vertices']
    neighbor_1_vertex_0 = neighbor_1[0]
    neighbor_1_vertex_1 = neighbor_1[1]
    neighbor_1_length = np.linalg.norm(neighbor_1_vertex_0 - neighbor_1_vertex_1)
    neighbor_1_features = [neighbor_1_vertex_0[0], neighbor_1_vertex_0[1], neighbor_1_vertex_0[2], neighbor_1_vertex_1[0], neighbor_1_vertex_1[1], neighbor_1_vertex_1[2], neighbor_1_length]
    if i == 0: f0_n1_features = neighbor_1_features
    elif i == 1: f1_n1_features = neighbor_1_features
    
  return np.array([edge_features, f0_n0_features, f0_n1_features, f1_n0_features, f1_n1_features])
  
  
  
def compute_edges_features(edges_properties, features_type):
  if features_type == 'meshcnn_like_features':
    edges_features = np.array([[d['dihaedral_angle'], d['inner_angle_1'], d['inner_angle_2'], d['length_height_ratio_1'], d['length_height_ratio_2']] for d in edges_properties])
  elif features_type == 'simple_features':
    edges_features = np.array([simple_features(d) for d in edges_properties])
  elif features_type == 'simple_features_2d':
    edges_features = np.zeros((1, 5, 7))
    for edge_properties in edges_properties:
      edge_features = simple_features_2d(edge_properties)
      edge_features = np.expand_dims(edge_features, axis=0) 
      edges_features = np.vstack((edges_features, edge_features))
  
  return edges_features    

def compute_edges_properties(mesh):
  vertices = mesh.vertices
  faces = mesh.faces

  edge_to_faces = {}
  for face in faces:
      face_edges = [(face[i], face[(i + 1) % 3]) for i in range(3)]
      face_edges = [tuple(sorted(edge)) for edge in face_edges]
      for edge in face_edges:
          if edge not in edge_to_faces:
              edge_to_faces[edge] = []
          edge_to_faces[edge].append(face)
          
  #* Compute the properties of the edges (vertices, neighbors, etc.)
  edges_properties = []
  for edge, faces_containing_edge in edge_to_faces.items():
    edge_properties = {
      'edge': edge,
      'vertices': [],
      'face_0': 
      { 
        'vertices': [],
        'neighbor_0': { 'edge': [], 'vertices': [] }, 
        'neighbor_1': { 'edge': [], 'vertices': [] }
      },
      'face_1': 
      { 
        'vertices': [],
        'neighbor_0': { 'edge': [], 'vertices': [] }, 
        'neighbor_1': { 'edge': [], 'vertices': [] }
      },
    }
  
    edge_properties['vertices'] = [vertices[list(edge)[0]], vertices[list(edge)[1]]]    
    for i, face in enumerate(faces_containing_edge):
      
      face_vertices = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
      
      neighbors_edges = [tuple(sorted((face[i], face[(i + 1) % 3]))) for i in range(3) if tuple(sorted((face[i], face[(i + 1) % 3]))) != edge]
      neighbor_edge_0 = neighbors_edges[0]
      neighbor_edge_1 = neighbors_edges[1]
      
      neighbors_edge_vertices = [vertices[list(e)] for e in neighbors_edges]
      neighbor_edge_vertices_0 = neighbors_edge_vertices[0]
      neighbor_edge_vertices_1 = neighbors_edge_vertices[1]

      edge_properties[f"face_{i}"]['vertices'] = face_vertices
      edge_properties[f"face_{i}"]['neighbor_0']['edge'] = neighbor_edge_0
      edge_properties[f"face_{i}"]['neighbor_0']['vertices'] = neighbor_edge_vertices_0
      edge_properties[f"face_{i}"]['neighbor_1']['edge'] = neighbor_edge_1
      edge_properties[f"face_{i}"]['neighbor_1']['vertices'] = neighbor_edge_vertices_1
      
    edges_properties.append(edge_properties)
    
  return edges_properties



if __name__ == '__main__':  
  dataset_path = 'datasets/human/'
  processed_data_path = 'datasets/simple_edge_features_2d/--'
  features_type = 'meshcnn_like_features'

  for subfolder in os.listdir(dataset_path):
    subfolder_path = os.path.join(dataset_path, subfolder)  
    if os.path.isdir(subfolder_path):
      print(f"Processing subfolder {subfolder} ...")
      mesh_files = [f for f in os.listdir(subfolder_path) if f.endswith('.obj')]
      
      for mesh_file in tqdm(mesh_files, desc=f'Processing meshes in {subfolder}'):
        if mesh_file.endswith('.obj'):
          mesh_path = os.path.join(subfolder_path, mesh_file)
          mesh = trimesh.load(mesh_path)
          edges_properties = compute_edges_properties(mesh)
          edges_features = compute_edges_features(edges_properties, features_type)
          edges_features_tensor = torch.tensor(edges_features)
          
          if not os.path.exists(os.path.join(processed_data_path, subfolder)):
            os.makedirs(os.path.join(processed_data_path, subfolder))
            
          processed_data_file = os.path.join(processed_data_path, subfolder, f"edges_features_{mesh_file.split('.')[0]}.pt")
          torch.save(edges_features_tensor, processed_data_file)