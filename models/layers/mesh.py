from tempfile import mkstemp
from shutil import move
import torch
import numpy as np
import os
from models.layers.mesh_union import MeshUnion
from models.layers.mesh_prepare import fill_mesh


class Mesh:

    def __init__(self, file=None, num_aug=1, scale_verts=None, flip_edges=None, slide_verts=None):

        self.vs = self.v_mask = self.filename = self.features = self.edge_areas = None
        self.edges = self.gemm_edges = self.sides = None
        self.pool_count = 0
        
        fill_mesh(self, file, num_aug=num_aug, scale_verts=scale_verts, flip_edges=flip_edges, slide_verts=slide_verts)

    def extract_features(self):
        return self.features
    
    def get_edge_areas(self):
        return self.edge_areas

    def merge_vertices(self, edge_id):
        self.remove_edge(edge_id)
        edge = self.edges[edge_id]
        v_a = self.vs[edge[0]]
        v_b = self.vs[edge[1]]
        v_a.__iadd__(v_b)
        v_a.__itruediv__(2)
        self.v_mask[edge[1]] = False
        mask = self.edges == edge[1]
        self.ve[edge[0]].extend(self.ve[edge[1]])
        self.edges[mask] = edge[0]

    def remove_vertex(self, v):
        self.v_mask[v] = False

    def remove_edge(self, edge_id):
        vs = self.edges[edge_id]
        for v in vs:
            if edge_id not in self.ve[v]:
                print(self.ve[v])
                print(self.filename)
            self.ve[v].remove(edge_id)