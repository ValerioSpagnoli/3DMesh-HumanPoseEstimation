import os
import trimesh
import torch
from tqdm import tqdm

def batch_edge_features(edges_properties, device):
    num_edges = len(edges_properties)
    
    edge_vertices_0 = torch.zeros((num_edges, 3), device=device)
    edge_vertices_1 = torch.zeros((num_edges, 3), device=device)
    f0_n0_v0 = torch.zeros((num_edges, 3), device=device)
    f0_n0_v1 = torch.zeros((num_edges, 3), device=device)
    f0_n1_v0 = torch.zeros((num_edges, 3), device=device)
    f0_n1_v1 = torch.zeros((num_edges, 3), device=device)
    
    f1_n0_v0 = torch.zeros((num_edges, 3), device=device)
    f1_n0_v1 = torch.zeros((num_edges, 3), device=device)
    f1_n1_v0 = torch.zeros((num_edges, 3), device=device)
    f1_n1_v1 = torch.zeros((num_edges, 3), device=device)
    
    for i, edge_properties in enumerate(edges_properties):
        edge_vertices_0[i] = torch.tensor(edge_properties['vertices'][0], device=device)
        edge_vertices_1[i] = torch.tensor(edge_properties['vertices'][1], device=device)

        f0_n0_v0[i] = torch.tensor(edge_properties['face_0']['neighbor_0']['vertices'][0], device=device)
        f0_n0_v1[i] = torch.tensor(edge_properties['face_0']['neighbor_0']['vertices'][1], device=device)
        f0_n1_v0[i] = torch.tensor(edge_properties['face_0']['neighbor_1']['vertices'][0], device=device)
        f0_n1_v1[i] = torch.tensor(edge_properties['face_0']['neighbor_1']['vertices'][1], device=device)
        
        f1_n0_v0[i] = torch.tensor(edge_properties['face_1']['neighbor_0']['vertices'][0], device=device)
        f1_n0_v1[i] = torch.tensor(edge_properties['face_1']['neighbor_0']['vertices'][1], device=device)
        f1_n1_v0[i] = torch.tensor(edge_properties['face_1']['neighbor_1']['vertices'][0], device=device)
        f1_n1_v1[i] = torch.tensor(edge_properties['face_1']['neighbor_1']['vertices'][1], device=device)
        
    edge_length = torch.norm(edge_vertices_0 - edge_vertices_1, dim=1, keepdim=True)
    f0_n0_length = torch.norm(f0_n0_v0 - f0_n0_v1, dim=1, keepdim=True)
    f0_n1_length = torch.norm(f0_n1_v0 - f0_n1_v1, dim=1, keepdim=True)
    f1_n0_length = torch.norm(f1_n0_v0 - f1_n0_v1, dim=1, keepdim=True)
    f1_n1_length = torch.norm(f1_n1_v0 - f1_n1_v1, dim=1, keepdim=True)

    edge_features = torch.cat([edge_vertices_0, edge_vertices_1, edge_length], dim=1)
    f0_n0_features = torch.cat([f0_n0_v0, f0_n0_v1, f0_n0_length], dim=1)
    f0_n1_features = torch.cat([f0_n1_v0, f0_n1_v1, f0_n1_length], dim=1)
    f1_n0_features = torch.cat([f1_n0_v0, f1_n0_v1, f1_n0_length], dim=1)
    f1_n1_features = torch.cat([f1_n1_v0, f1_n1_v1, f1_n1_length], dim=1)
    
    return torch.stack([edge_features, f0_n0_features, f0_n1_features, f1_n0_features, f1_n1_features], dim=1)

def compute_edges_properties(mesh):
    vertices = mesh.vertices
    faces = mesh.faces

    edge_to_faces = {}
    for face in faces:
        face_edges = [tuple(sorted((face[i], face[(i + 1) % 3]))) for i in range(3)]
        for edge in face_edges:
            edge_to_faces.setdefault(edge, []).append(face)

    edges_properties = []
    for edge, faces_containing_edge in edge_to_faces.items():
        edge_properties = {
            'edge': edge,
            'vertices': [vertices[edge[0]], vertices[edge[1]]],
            'face_0': {'vertices': [], 'neighbor_0': {'vertices': []}, 'neighbor_1': {'vertices': []}},
            'face_1': {'vertices': [], 'neighbor_0': {'vertices': []}, 'neighbor_1': {'vertices': []}}
        }

        for i, face in enumerate(faces_containing_edge):
            neighbors_edges = [tuple(sorted((face[j], face[(j + 1) % 3]))) for j in range(3) if tuple(sorted((face[j], face[(j + 1) % 3]))) != edge]
            edge_properties[f'face_{i}']['vertices'] = [vertices[face[0]], vertices[face[1]], vertices[face[2]]]
            edge_properties[f'face_{i}']['neighbor_0']['vertices'] = [vertices[neighbors_edges[0][0]], vertices[neighbors_edges[0][1]]]
            edge_properties[f'face_{i}']['neighbor_1']['vertices'] = [vertices[neighbors_edges[1][0]], vertices[neighbors_edges[1][1]]]

        edges_properties.append(edge_properties)
    
    return edges_properties

def extract_edge_features(mesh, normalize=True, device='cpu'):    
    if normalize:
        centroid = mesh.centroid 
        scale = mesh.scale   
        mesh.vertices -= centroid
        mesh.vertices /= scale
    
    edges_properties = compute_edges_properties(mesh)
    edges_features = batch_edge_features(edges_properties, device)
    
    return edges_features
    
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    split = 'test'
    features_type = 'simple_edge_features_2d'
    dataset_path = f'datasets/human_seg/mesh_{split}/'
    processed_data_path = f'datasets/human_seg/edge_features_{split}/'
    mesh_files = [f for f in os.listdir(dataset_path) if f.endswith('.obj')]

    for mesh_file in tqdm(mesh_files, desc=f'Processing meshes in {split}'):
        mesh_path = os.path.join(dataset_path, mesh_file)
        mesh = trimesh.load(mesh_path)
        edges_features = extract_edge_features(mesh, normalize=True, device=device)
        processed_data_file = os.path.join(processed_data_path, f"{mesh_file.split('.')[0]}.pt")
        torch.save(edges_features, processed_data_file)