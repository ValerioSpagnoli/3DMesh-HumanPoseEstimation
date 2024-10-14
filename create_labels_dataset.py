import os
import trimesh
import numpy as np
from pathlib import Path
import tqdm
from collections import defaultdict
import plotly.graph_objects as go
import trimesh
import pandas as pd


# classes: 
# 7 8 -> caviglia
# 6 7 -> ginocchio
# 5 6 -> anche
# 4 5 -> spalle
# 3 4 -> gomiti
# 2 3 -> polsi
# 1 5 -> collo

# 1 -> testa
# 2 -> mani
# 3 -> avambraccia
# 4 -> braccia
# 5 -> busto
# 6 -> gambe
# 7 -> tibie
# 8 -> piedi

mapping_class_name = {
  "{'7', '8'}": 'ankles',
  "{'8', '7'}": 'ankles',
  "{'6', '7'}": 'knees',
  "{'7', '6'}": 'knees',
  "{'5', '6'}": 'hips',
  "{'6', '5'}": 'hips',
  "{'4', '5'}": 'shoulders',
  "{'5', '4'}": 'shoulders',
  "{'3', '4'}": 'elbows',
  "{'4', '3'}": 'elbows',
  "{'2', '3'}": 'wrists',
  "{'3', '2'}": 'wrists',
  "{'1', '5'}": 'neck',
  "{'5', '1'}": 'neck' 
}

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


def save_keypoints_to_file(name, keypoints, classes):
    df = pd.DataFrame(keypoints, columns=['x', 'y', 'z'])
    if classes is not None: df['class'] = classes
    df.to_csv(name, index=False)
    
    


if __name__ == "__main__":
  root_dir = 'datasets/human_body_segmentation'
  mesh_path = os.path.join(root_dir, f'meshes')
  seg_path = os.path.join(root_dir, f'seg')
  output_keypoints_path = os.path.join(root_dir, f'keypoints')

  mesh_files = [f for f in os.listdir(mesh_path) if f.endswith('.obj')]
  seg_files = [f for f in os.listdir(seg_path) if f.endswith('.eseg')]
  mesh_files.sort()
  seg_files.sort()
  pair_list = list(zip(mesh_files, seg_files))

  m = 0
  count_num_keypoints = {}
  for mesh_file, seg_file in pair_list:

    edge_label_pairs = create_labels_for_mesh(mesh_path, seg_path, mesh_file, seg_file)
    useful_vertices_dict = create_useful_vertices_dict(edge_label_pairs)
    useful_pairs = [elem for elem in edge_label_pairs if elem[0][0] in useful_vertices_dict and elem[0][1] in useful_vertices_dict]    
    connected_components = find_connected_components(useful_pairs)
    
    connected_components_classes = {}
    for i, connected_component in enumerate(connected_components):
      classes = set()
      for (v1, v2), cls in useful_pairs:
        if v1 in connected_component and v2 in connected_component:
          classes.add(cls)
      connected_components_classes[i] = classes
              
    already_deleted = False
    for i, connected_component in enumerate(connected_components):
      if already_deleted: break
      if connected_components_classes[i] == {'5', '6'}:
        for j, connected_component2 in enumerate(connected_components):
          if connected_components_classes[j] == {'5', '6'} and i != j:
            connected_components[i] = connected_components[i].union(connected_components[j])
            connected_components.pop(j)
            already_deleted = True
            break
      
    # keypoint extraction
    mesh = trimesh.load_mesh(os.path.join(mesh_path, mesh_file))
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

    
    # if len(keypoints) == 12:
    #   keypoints_path = os.path.join(output_keypoints_path, mesh_file[:-4] + '.csv')
    #   save_keypoints_to_file(keypoints_path, keypoints, None)
    # else:
    #   os.rename(os.path.join(mesh_path, mesh_file), os.path.join(root_dir, 'other_mesh', mesh_file))
      
    count_num_keypoints[len(keypoints)] = count_num_keypoints.get(len(keypoints), 0) + 1
  
    #* Plot
    fig = go.Figure()
    fig.update_layout(scene=dict(aspectmode='data'))
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown', 'grey', 'black', 'white', 'silver']

    # Plot the mesh
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1],  z=vertices[:, 2], i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2], color='lightgrey', opacity=0.5))

    # Plot keypoints
    for keypoint in keypoints:
        fig.add_trace(go.Scatter3d(x=[keypoint[0]], y=[keypoint[1]], z=[keypoint[2]], mode='markers', marker=dict(size=3, color='green')))
        pass

    # Plot connected components of a specific class
    # cls = {'1', '5'}
    # for i, connected_component_vertices in enumerate(connected_components_vertices):
    #   if connected_components_classes[i] == cls:
    #     x = [vertex[0] for vertex in connected_component_vertices]
    #     y = [vertex[1] for vertex in connected_component_vertices]
    #     z = [vertex[2] for vertex in connected_component_vertices]
    #     fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=colors[i])))
    
    # Plot useful vertices
    useful_vertices_coordinates = [vertices[vertex] for vertex in useful_vertices_dict]
    for useful_vertex in useful_vertices_coordinates:
        #fig.add_trace(go.Scatter3d(x=[useful_vertex[0]], y=[useful_vertex[1]], z=[useful_vertex[2]], mode='markers', marker=dict(size=3)))
        pass

    # Plot connected components
    for i, connected_component_vertices in enumerate(connected_components_vertices):
      x = [vertex[0] for vertex in connected_component_vertices]
      y = [vertex[1] for vertex in connected_component_vertices]
      z = [vertex[2] for vertex in connected_component_vertices]
      # fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', marker=dict(size=3, color=colors[i])))

    # Plot useful_pairs edges
    for pair in useful_pairs:
      edge = pair[0]
      cls = pair[1]
      v0 = vertices[edge[0]]
      v1 = vertices[edge[1]]
      #fig.add_trace(go.Scatter3d(x=[v0[0], v1[0]], y=[v0[1], v1[1]], z=[v0[2], v1[2]], mode='lines', line=dict(width=4, color=colors[int(cls)])))

    # Show  0, 20 
    fig.show()
    break
    if m==0:  
      fig.show()
      break
    m += 1
        
  print(count_num_keypoints)