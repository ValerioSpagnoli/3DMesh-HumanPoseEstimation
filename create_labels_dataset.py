import os
import trimesh
import numpy as np
from pathlib import Path
import tqdm
from collections import defaultdict
import plotly.graph_objects as go
import trimesh

# Find EDGES starting from the faces of the mesh
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
    dizionario_vertici = {}

    for (vertice1, vertice2), classe in lati:
        if vertice1 not in dizionario_vertici:
            dizionario_vertici[vertice1] = set(classe)
        else:
            dizionario_vertici[vertice1].add(classe)
        if vertice2 not in dizionario_vertici:
            dizionario_vertici[vertice2] = set(classe)
        else:
            dizionario_vertici[vertice2].add(classe)

    final_dict = {}
    for key, value in dizionario_vertici.items():
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
  # root_dir = 'datasets/human_seg'
  # train_path = os.path.join(root_dir, 'train')
  # seg_path = os.path.join(root_dir, 'seg_train')

  # mesh_files = [f for f in os.listdir(train_path) if f.endswith('.obj')]
  # mesh_files.sort()
  # seg_files = [f for f in os.listdir(seg_path) if f.endswith('.eseg')]
  # seg_files.sort()
  # pair_list = list(zip(mesh_files, seg_files))
  
  # z = 0
  # for mesh_file, seg_file in pair_list:
    
  #   edge_label_pairs = create_labels_for_mesh(mesh_file, seg_file)
  #   useful_vertices_dict = create_useful_vertices_dict(edge_label_pairs)
  #   useful_pairs = [elem for elem in edge_label_pairs if elem[0][0] in useful_vertices_dict or elem[0][1] in useful_vertices_dict]
  #   connected_components = find_connected_components(useful_pairs)

  root_dir = 'datasets/human_seg'
  train_path = os.path.join(root_dir, 'train')
  seg_path = os.path.join(root_dir, 'seg_train')

  mesh_files = [f for f in os.listdir(train_path) if f.endswith('.obj')]
  mesh_files.sort()
  seg_files = [f for f in os.listdir(seg_path) if f.endswith('.eseg')]
  seg_files.sort()
  pair_list = list(zip(mesh_files, seg_files))

  count_num_keypoints = {}
  i = 0
  for mesh_file, seg_file in pair_list:

    edge_label_pairs = create_labels_for_mesh(train_path, seg_path, mesh_file, seg_file)
    all_labels = set()
    for edge_label_pair in edge_label_pairs:
      all_labels.add(edge_label_pair[1])

    useful_vertices_dict = create_useful_vertices_dict(edge_label_pairs)

    # useful_pairs = [elem for elem in edge_label_pairs if elem[0][0] in useful_vertices_dict or elem[0][1] in useful_vertices_dict]
    useful_pairs = []
    for elem in edge_label_pairs:
      if elem[0][0] in useful_vertices_dict and elem[0][1] in useful_vertices_dict:
        useful_pairs.append(elem)
    
    connected_components = find_connected_components(useful_pairs)

    # using useful vertices dict compute all the classes of a single connected component
    connected_components_classes = []
    for connected_component in connected_components:
      connected_component_classes = set()
      for vertex in connected_component:
        connected_component_classes = connected_component_classes.union(useful_vertices_dict[vertex])
      connected_components_classes.append(connected_component_classes)

    

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
      # compute the mean of the connected component
      mean = [0, 0, 0]
      for vertex in connected_component_vertices:
        mean[0] += vertex[0]
        mean[1] += vertex[1]
        mean[2] += vertex[2]
      mean[0] /= len(connected_component_vertices)
      mean[1] /= len(connected_component_vertices)
      mean[2] /= len(connected_component_vertices)
      keypoints.append(mean)

    print(len(keypoints))
    count_num_keypoints[len(keypoints)] = count_num_keypoints.get(len(keypoints), 0) + 1
  

    useful_vertices_coordinates = []
    for vertex in useful_vertices_dict:
      useful_vertices_coordinates.append(vertices[vertex])

    

    fig = go.Figure()

    # Add the mesh data
    fig.add_trace(go.Mesh3d(
        x=vertices[:, 0], 
        y=vertices[:, 1], 
        z=vertices[:, 2], 
        i=mesh.faces[:, 0], 
        j=mesh.faces[:, 1], 
        k=mesh.faces[:, 2], 
        opacity=0.5
    ))


    for keypoint in keypoints:
        fig.add_trace(go.Scatter3d(
            x=[keypoint[0]], 
            y=[keypoint[1]], 
            z=[keypoint[2]], 
            mode='markers', 
            marker=dict(size=5)
        ))

    # #plot the useful vertices
    # # for useful_vertex in useful_vertices_coordinates:
    # #     fig.add_trace(go.Scatter3d(
    # #         x=[useful_vertex[0]], 
    # #         y=[useful_vertex[1]], 
    # #         z=[useful_vertex[2]], 
    # #         mode='markers', 
    # #         marker=dict(size=3)
    # #     ))

    # # plot connected components vertices with different colors (same color for the same connected component)
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange', 'pink', 'cyan', 'magenta', 'brown']
    # # for i, connected_component_vertices in enumerate(connected_components_vertices):
    # #   x = []
    # #   y = []
    # #   z = []
    # #   for vertex in connected_component_vertices:
    # #     x.append(vertex[0])
    # #     y.append(vertex[1])
    # #     z.append(vertex[2])
    # #   fig.add_trace(go.Scatter3d(
    # #       x=x, 
    # #       y=y, 
    # #       z=z, 
    # #       mode='markers', 
    # #       marker=dict(size=3, color=colors[i])
    # #   ))

    # # plot useful_pairs edges using one color for each class
    for pair in useful_pairs:
      edge = pair[0]
      cls = pair[1]
      v0 = vertices[edge[0]]
      v1 = vertices[edge[1]]
      fig.add_trace(go.Scatter3d(
          x=[v0[0], v1[0]], 
          y=[v0[1], v1[1]], 
          z=[v0[2], v1[2]], 
          mode='lines', 
          line=dict(width=2, color=colors[int(cls)])
      ))


    # # Set the aspect ratio to ensure correct proportions
    fig.update_layout(
        scene=dict(
            aspectmode='data'
        )
    )

    # Show the plot
    if len(keypoints) == 11:
      fig.show()

  print(count_num_keypoints)