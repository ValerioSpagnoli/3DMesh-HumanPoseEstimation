import os
import trimesh
import numpy as np
from pathlib import Path
import tqdm
from collections import defaultdict
import plotly.graph_objects as go
import trimesh

def get_edge_faces(faces):
    edge_count = 0
    edge_faces = []
    edge2keys = dict()
    for face_id, face in enumerate(faces):
        for i in range(3):
            cur_edge = tuple(sorted((face[i], face[(i + 1) % 3])))
            if cur_edge not in edge2keys:
                edge2keys[cur_edge] = edge_count
                edge_count += 1
                edge_faces.append(np.array([cur_edge[0], cur_edge[1], -1, -1]))
            edge_key = edge2keys[cur_edge]
            if edge_faces[edge_key][2] == -1:
                edge_faces[edge_key][2] = face_id
            else:
                edge_faces[edge_key][3] = face_id
    return edge_count, np.array(edge_faces), edge2keys

def create_labels_for_mesh(train_path, seg_path, mesh_file, seg_file):
    mesh = trimesh.load(os.path.join(train_path, mesh_file))
    edge_count, edge_faces, edge2keys = get_edge_faces(mesh.faces)
    seg_list = [line.rstrip() for line in open(os.path.join(seg_path, seg_file), "r")]
    edge_label_pair = list(zip(edge2keys, seg_list))
    return edge_label_pair

def create_useful_vertices_dict(lati):
    vertices_dict = {}

    for (vertex1, vertex2), cls in lati:
        if vertex1 not in vertices_dict:
            vertices_dict[vertex1] = set(cls)
        else:
            vertices_dict[vertex1].add(cls)
        if vertex2 not in vertices_dict:
            vertices_dict[vertex2] = set(cls)
        else:
            vertices_dict[vertex2].add(cls)

    final_dict = {}
    for key, value in vertices_dict.items():
        if len(value) > 1:
            final_dict[key] = value
    return final_dict

def delete_useless_vertices_from_list(edge_label_pairs, useful_vertices_dict):
    for (v1,v2), label in edge_label_pairs:
        if v1 not in useful_vertices_dict and v2 not in useful_vertices_dict:
            edge_label_pairs.remove(((v1,v2),label))

def find_connected_components(useful_pairs):
    adj_list = defaultdict(list)
    for (v1, v2), _ in useful_pairs:
        adj_list[v1].append(v2)
        adj_list[v2].append(v1)

    connected_components = []
    i = 0
    while i < len(adj_list):
      key1 = list(adj_list.keys())[i]
      value1 = adj_list[key1]
      j = 0
      while j < len(value1):
        if key1 == value1[j]: 
          j += 1
          continue
        value2 = adj_list[value1[j]]
        for elem in value2:
          if elem not in value1:
            value1.append(elem)
        j += 1
      
      connected_component = set(value1) 
      connected_components.append(connected_component)
      
      deleted_key = False
      for elem in connected_component:
        if elem == key1:
          deleted_key = True
        adj_list.pop(elem)
      
      if not deleted_key: i += 1

    return connected_components



if __name__ == "__main__":
  root_dir = 'datasets/human_seg'
  train_path = os.path.join(root_dir, 'train')
  seg_path = os.path.join(root_dir, 'seg_train')

  mesh_files = [f for f in os.listdir(train_path) if f.endswith('.obj')]
  seg_files = [f for f in os.listdir(seg_path) if f.endswith('.eseg')]
  mesh_files.sort()
  seg_files.sort()
  pair_list = list(zip(mesh_files, seg_files))

  count_num_keypoints = {}
  for mesh_file, seg_file in pair_list:

    edge_label_pairs = create_labels_for_mesh(train_path, seg_path, mesh_file, seg_file)
    useful_vertices_dict = create_useful_vertices_dict(edge_label_pairs)
    useful_pairs = [elem for elem in edge_label_pairs if elem[0][0] in useful_vertices_dict and elem[0][1] in useful_vertices_dict]    
    connected_components = find_connected_components(useful_pairs)

    # keypoint extraction
    mesh = trimesh.load_mesh(os.path.join(train_path, mesh_file))
    vertices = mesh.vertices
    connected_components_vertices = []
    for connected_component in connected_components:
      connected_component_vertices = set()
      for vertex in connected_component:
        connected_component_vertices.add(vertices[vertex])   
      connected_components_vertices.append(connected_component_vertices)

    keypoints = []
    for connected_component_vertices in connected_components_vertices:
      mean = [0, 0, 0]
      for vertex in connected_component_vertices:
        mean[0] += vertex[0]
        mean[1] += vertex[1]
        mean[2] += vertex[2]
      mean[0] /= len(connected_component_vertices)
      mean[1] /= len(connected_component_vertices)
      mean[2] /= len(connected_component_vertices)
      keypoints.append(mean)

    count_num_keypoints[len(keypoints)] = count_num_keypoints.get(len(keypoints), 0) + 1
  
    #* Plot
    fig = go.Figure()
    fig.update_layout(scene=dict(aspectmode='data'))
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown', 'grey', 'black', 'white', 'silver']

    # Plot the mesh
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1],  z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], opacity=0.5))

    # Plot keypoints
    for keypoint in keypoints:
        fig.add_trace(go.Scatter3d(x=[keypoint[0]], y=[keypoint[1]], z=[keypoint[2]], mode='markers', marker=dict(size=5)))

    # Plot useful vertices
    useful_vertices_coordinates = [vertices[vertex] for vertex in useful_vertices_dict]
    for useful_vertex in useful_vertices_coordinates:
        fig.add_trace(go.Scatter3d(x=[useful_vertex[0]], y=[useful_vertex[1]], z=[useful_vertex[2]], mode='markers', marker=dict(size=3)))

    # Plot connected components
    for i, connected_component_vertices in enumerate(connected_components_vertices):
      x = [vertex[0] for vertex in connected_component_vertices]
      y = [vertex[1] for vertex in connected_component_vertices]
      z = [vertex[2] for vertex in connected_component_vertices]
      fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=colors[i])))

    # Plot useful_pairs edges
    for pair in useful_pairs:
      edge = pair[0]
      cls = pair[1]
      v0 = vertices[edge[0]]
      v1 = vertices[edge[1]]
      fig.add_trace(go.Scatter3d(x=[v0[0], v1[0]], y=[v0[1], v1[1]], z=[v0[2], v1[2]], mode='lines', line=dict(width=2, color=colors[int(cls)])))

    # Show  
    # fig.show()
      
  print(count_num_keypoints)