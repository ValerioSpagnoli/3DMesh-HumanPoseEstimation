import os
import trimesh
import torch
from tqdm import tqdm

import torch
import numpy as np

def batch_meshcnn_edge_features(edge_properties_list, device):
    batch_size = len(edge_properties_list)

    # Initialize tensors to hold the features for all edges in the batch
    dihedral_angles = torch.zeros(batch_size, device=device)
    inner_angles_1 = torch.zeros(batch_size, device=device)
    inner_angles_2 = torch.zeros(batch_size, device=device)
    length_height_ratios_1 = torch.zeros(batch_size, device=device)
    length_height_ratios_2 = torch.zeros(batch_size, device=device)

    for idx, edge_properties in enumerate(edge_properties_list):
        # Compute the dihedral angle
        face_0_vertices = torch.tensor(np.array(edge_properties['face_0']['vertices']), device=device)
        face_1_vertices = torch.tensor(np.array(edge_properties['face_1']['vertices']), device=device)

        # Compute normals for both faces
        normal_0 = torch.linalg.cross(face_0_vertices[1] - face_0_vertices[0], face_0_vertices[2] - face_0_vertices[0])
        normal_0 /= torch.norm(normal_0)

        normal_1 = torch.linalg.cross(face_1_vertices[1] - face_1_vertices[0], face_1_vertices[2] - face_1_vertices[0])
        normal_1 /= torch.norm(normal_1)

        # Compute cosine of the dihedral angle
        cos_angle = torch.clamp(torch.dot(normal_0, normal_1), -1, 1)
        dihedral_angles[idx] = torch.acos(cos_angle)

        # Compute inner angles
        for i in range(2):
            edge_0 = torch.tensor(np.array(edge_properties[f'face_{i}']['neighbor_0']['vertices']), device=device)
            edge_1 = torch.tensor(np.array(edge_properties[f'face_{i}']['neighbor_1']['vertices']), device=device)
            common_vertex = (face_0_vertices if i == 0 else face_1_vertices)[0]  # Take first vertex as common

            vertex_1 = edge_0[~torch.all(edge_0 == common_vertex, dim=1)][0]
            vertex_2 = edge_1[~torch.all(edge_1 == common_vertex, dim=1)][0]

            d1 = common_vertex - vertex_1
            d1 /= torch.norm(d1)
            d2 = vertex_2 - common_vertex
            d2 /= torch.norm(d2)

            cos_angle_inner = torch.clamp(torch.dot(d1, d2), -1, 1)
            if cos_angle_inner.item() != 1:  # Prevent potential NaN from acos(1)
                inner_angle = torch.acos(cos_angle_inner)
            else:
                inner_angle = torch.tensor(0.0, device=device)  # Handle edge case where angles are 0
            if i == 0:
                inner_angles_1[idx] = inner_angle
            else:
                inner_angles_2[idx] = inner_angle

        # Compute the length-height ratio
        edge_length = torch.linalg.norm(torch.tensor(edge_properties['vertices'][0]) - torch.tensor(edge_properties['vertices'][1]))
        for i in range(2):
            face_vertex_0 = torch.tensor(edge_properties[f'face_{i}']['vertices'][0], device=device)
            face_vertex_1 = torch.tensor(edge_properties[f'face_{i}']['vertices'][1], device=device)
            face_vertex_2 = torch.tensor(edge_properties[f'face_{i}']['vertices'][2], device=device)

            face_area = torch.norm(torch.linalg.cross(face_vertex_1 - face_vertex_0, face_vertex_2 - face_vertex_0))
            length_height_ratio = (edge_length * edge_length) / (2 * face_area)
            if i == 0:
                length_height_ratios_1[idx] = length_height_ratio
            else:
                length_height_ratios_2[idx] = length_height_ratio

    # Package the results in a dictionary
    edge_features_batch = {
        'dihedral_angle': dihedral_angles,
        'inner_angle_1': inner_angles_1,
        'inner_angle_2': inner_angles_2,
        'length_height_ratio_1': length_height_ratios_1,
        'length_height_ratio_2': length_height_ratios_2
    }

    return edge_features_batch


def batch_simple_edge_features(edges_properties, device):
    num_edges = len(edges_properties)

    edge_vertices_0 = torch.zeros((num_edges, 3), dtype=torch.float32, device=device)
    edge_vertices_1 = torch.zeros((num_edges, 3), dtype=torch.float32, device=device)

    for i, edge_properties in enumerate(edges_properties):
        edge_vertices_0[i] = torch.tensor(edge_properties['vertices'][0], dtype=torch.float32, device=device)
        edge_vertices_1[i] = torch.tensor(edge_properties['vertices'][1], dtype=torch.float32, device=device)

    edge_lengths = torch.norm(edge_vertices_0 - edge_vertices_1, dim=1, keepdim=True)

    features = torch.cat([edge_vertices_0, edge_vertices_1, edge_lengths], dim=1)
    return features

def batch_simple_edge_features_2d(edges_properties, device):
    num_edges = len(edges_properties)

    # Pre-allocate tensors for all edges and their neighbors
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
    
    # Fill tensors with edge and neighbor vertices
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
        
    # Compute edge lengths in batch
    edge_length = torch.norm(edge_vertices_0 - edge_vertices_1, dim=1, keepdim=True)
    f0_n0_length = torch.norm(f0_n0_v0 - f0_n0_v1, dim=1, keepdim=True)
    f0_n1_length = torch.norm(f0_n1_v0 - f0_n1_v1, dim=1, keepdim=True)
    f1_n0_length = torch.norm(f1_n0_v0 - f1_n0_v1, dim=1, keepdim=True)
    f1_n1_length = torch.norm(f1_n1_v0 - f1_n1_v1, dim=1, keepdim=True)

    # Concatenate the features in a batch
    edge_features = torch.cat([edge_vertices_0, edge_vertices_1, edge_length], dim=1)
    f0_n0_features = torch.cat([f0_n0_v0, f0_n0_v1, f0_n0_length], dim=1)
    f0_n1_features = torch.cat([f0_n1_v0, f0_n1_v1, f0_n1_length], dim=1)
    f1_n0_features = torch.cat([f1_n0_v0, f1_n0_v1, f1_n0_length], dim=1)
    f1_n1_features = torch.cat([f1_n1_v0, f1_n1_v1, f1_n1_length], dim=1)
    
    # Stack all features into a single tensor [num_edges, 3 (features), 7 (dimensions)]
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


if __name__ == '__main__':
    features_type = 'simple_edge_features_2d'
    dataset_path = 'datasets/human_seg/'
    processed_data_path = f'datasets/human_seg/preprocessed_train/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for subfolder in os.listdir(dataset_path):
        subfolder_path = os.path.join(dataset_path, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder {subfolder} ...")
            mesh_files = [f for f in os.listdir(subfolder_path) if f.endswith('.obj')]

            for mesh_file in tqdm(mesh_files, desc=f'Processing meshes in {subfolder}'):
                mesh_path = os.path.join(subfolder_path, mesh_file)
                mesh = trimesh.load(mesh_path)
                edges_properties = compute_edges_properties(mesh)

                # Use batch processing for edge features
                if features_type == 'simple_edge_features':
                    edges_features = batch_simple_edge_features(edges_properties, device)
                elif features_type == 'simple_edge_features_2d':
                  edges_features = batch_simple_edge_features_2d(edges_properties, device)
                elif features_type == 'meshcnn_edge_features':
                    edges_features = batch_meshcnn_edge_features(edges_properties, device)
                
                subfolder_processed_path = os.path.join(processed_data_path, subfolder)
                os.makedirs(subfolder_processed_path, exist_ok=True)

                processed_data_file = os.path.join(subfolder_processed_path, f"edges_features_{mesh_file.split('.')[0]}.pt")
                torch.save(edges_features, processed_data_file)
