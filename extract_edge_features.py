import torch


def batch_edge_features(edges_properties, device):
    num_edges = len(edges_properties)
    
    edge_vertices_0 = torch.zeros((num_edges, 3), device=device)
    edge_vertices_1 = torch.zeros((num_edges, 3), device=device)
    
    f0_n0_v0 = torch.zeros((num_edges, 3), device=device) # a v0
    f0_n0_v1 = torch.zeros((num_edges, 3), device=device) # a v1
    
    f0_n1_v0 = torch.zeros((num_edges, 3), device=device) # b v0
    f0_n1_v1 = torch.zeros((num_edges, 3), device=device) # b v1
    
    f1_n0_v0 = torch.zeros((num_edges, 3), device=device) # c v0
    f1_n0_v1 = torch.zeros((num_edges, 3), device=device) # c v1
    
    f1_n1_v0 = torch.zeros((num_edges, 3), device=device) # d v0
    f1_n1_v1 = torch.zeros((num_edges, 3), device=device) # d v1
        
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


    f1_n0_is_c = torch.logical_not(torch.any(torch.eq(f0_n0_v0, f1_n0_v0), dim=2) | torch.any(torch.eq(f0_n0_v1, f1_n0_v0), dim=2)) & \
                 torch.logical_not(torch.any(torch.eq(f0_n0_v0, f1_n0_v1), dim=2) | torch.any(torch.eq(f0_n0_v1, f1_n0_v1), dim=2))
    
    c0 = torch.where(f1_n0_is_c.unsqueeze(-1), f1_n0_v0, f1_n1_v0)
    c1 = torch.where(f1_n0_is_c.unsqueeze(-1), f1_n0_v1, f1_n1_v1)
    d0 = torch.where(f1_n0_is_c.unsqueeze(-1), f1_n1_v0, f1_n0_v0)
    d1 = torch.where(f1_n0_is_c.unsqueeze(-1), f1_n1_v1, f1_n0_v1)
    
    neighbors = {
        'a_v0': f0_n0_v0,
        'a_v1': f0_n0_v1,
        'b_v0': f0_n1_v0,
        'b_v1': f0_n1_v1,
        'c_v0': c0,
        'c_v1': c1,
        'd_v0': d0,
        'd_v1': d1
    }
    
    e1_v0 = torch.zeros((num_edges, 3), device=device) # e1 = |a - c|
    e1_v1 = torch.zeros((num_edges, 3), device=device) # e1 = |a - c|
    e1_v0 = torch.abs(neighbors['a_v0'] - neighbors['c_v0'])
    e1_v1 = torch.abs(neighbors['a_v1'] - neighbors['c_v1'])
    
    e2_v0 = torch.zeros((num_edges, 3), device=device) # e2 =  a + c 
    e2_v1 = torch.zeros((num_edges, 3), device=device) # e2 =  a + c
    e2_v0 = neighbors['a_v0'] + neighbors['c_v0']
    e2_v1 = neighbors['a_v1'] + neighbors['c_v1']
    
    e3_v0 = torch.zeros((num_edges, 3), device=device) # e3 = |b - d|
    e3_v1 = torch.zeros((num_edges, 3), device=device) # e3 = |b - d|
    e3_v0 = torch.abs(neighbors['b_v0'] - neighbors['d_v0'])
    e3_v1 = torch.abs(neighbors['b_v1'] - neighbors['d_v1'])
    
    e4_v0 = torch.zeros((num_edges, 3), device=device) # e4 =  b + d
    e4_v1 = torch.zeros((num_edges, 3), device=device) # e4 =  b + d
    e4_v0 = neighbors['b_v0'] + neighbors['d_v0']
    e4_v1 = neighbors['b_v1'] + neighbors['d_v1']
    
    
    e1_length = torch.norm(e1_v0 - e1_v1, dim=1, keepdim=True)
    e2_length = torch.norm(e2_v0 - e2_v1, dim=1, keepdim=True)
    e3_length = torch.norm(e3_v0 - e3_v1, dim=1, keepdim=True)
    e4_length = torch.norm(e4_v0 - e4_v1, dim=1, keepdim=True)
    
    e1_features = torch.cat([e1_v0, e1_v1, e1_length], dim=1)
    e2_features = torch.cat([e2_v0, e2_v1, e2_length], dim=1)
    e3_features = torch.cat([e3_v0, e3_v1, e3_length], dim=1)
    e4_features = torch.cat([e4_v0, e4_v1, e4_length], dim=1)
    

    # f0_n0_length = torch.norm(f0_n0_v0 - f0_n0_v1, dim=1, keepdim=True)
    # f0_n1_length = torch.norm(f0_n1_v0 - f0_n1_v1, dim=1, keepdim=True)
    # f1_n0_length = torch.norm(f1_n0_v0 - f1_n0_v1, dim=1, keepdim=True)
    # f1_n1_length = torch.norm(f1_n1_v0 - f1_n1_v1, dim=1, keepdim=True)

    # f0_n0_features = torch.cat([f0_n0_v0, f0_n0_v1, f0_n0_length], dim=1)
    # f0_n1_features = torch.cat([f0_n1_v0, f0_n1_v1, f0_n1_length], dim=1)
    # f1_n0_features = torch.cat([f1_n0_v0, f1_n0_v1, f1_n0_length], dim=1)
    # f1_n1_features = torch.cat([f1_n1_v0, f1_n1_v1, f1_n1_length], dim=1)
    

    edge_features = torch.cat([edge_vertices_0, edge_vertices_1, edge_length], dim=1)
    edge_length = torch.norm(edge_vertices_0 - edge_vertices_1, dim=1, keepdim=True)
    
    return torch.stack([edge_features, e1_features, e2_features, e3_features, e4_features], dim=1)

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

def extract_edge_features(mesh, device='cpu'):        
    edges_properties = compute_edges_properties(mesh)
    edges_features = batch_edge_features(edges_properties, device)
    
    return edges_features