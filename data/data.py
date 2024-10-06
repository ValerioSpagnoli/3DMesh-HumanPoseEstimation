import os
import numpy as np  
import pickle
from torch.utils import data
from utils.util import is_mesh_file, pad
from models.layers.mesh import Mesh

class ClassificationData(data.Dataset):

    def __init__(self, dataroot, phase, export_folder, device, ninput_edges, num_aug, scale_verts=None, flip_edges=None, slide_verts=None):
        super(ClassificationData, self).__init__()
      
        self.dataroot = dataroot  
        self.phase = phase  
        self.export_folder = export_folder
        self.device = device
        self.ninput_edges = ninput_edges
        self.num_aug = num_aug
        self.scale_verts = scale_verts
        self.flip_edges = flip_edges
        self.slide_verts = slide_verts
        
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        self.root = dataroot
        self.dir = os.path.join(dataroot)
        self.classes, self.class_to_idx = self.find_classes(self.dir)
        self.paths = self.make_dataset_by_class(self.dir, self.class_to_idx, phase)
        self.nclasses = len(self.classes)
        self.size = len(self.paths)
        self.get_mean_std()
        
    def get_nclasses(self):
        return self.nclasses
      
    def get_ninput_channels(self):
        return self.ninput_channels

    def __getitem__(self, index):
        path = self.paths[index][0]
        label = self.paths[index][1]
        mesh = Mesh(file=path, hold_history=False, export_folder=self.export_folder, num_aug=self.num_aug, scale_verts=self.scale_verts, flip_edges=self.flip_edges, slide_verts=self.slide_verts)
        meta = {'mesh': mesh, 'label': label}
        # get edge features
        edge_features = mesh.extract_features()
        edge_features = pad(edge_features, self.ninput_edges)
        meta['edge_features'] = (edge_features - self.mean) / self.std
        return meta

    def __len__(self):
        return self.size
      
    def get_mean_std(self):
      """ Computes Mean and Standard Deviation from Training Data
      If mean/std file doesn't exist, will compute one
      :returns
      mean: N-dimensional mean
      std: N-dimensional standard deviation
      ninput_channels: N
      (here N=5)
      """

      mean_std_cache = os.path.join(self.root, 'mean_std_cache.p')
      if not os.path.isfile(mean_std_cache):
          print('computing mean std from train data...')
          # doesn't run augmentation during m/std computation
          num_aug = self.num_aug
          self.num_aug = 1
          mean, std = np.array(0), np.array(0)
          for i, data in enumerate(self):
              if i % 500 == 0:
                  print('{} of {}'.format(i, self.size))
              features = data['edge_features']
              mean = mean + features.mean(axis=1)
              std = std + features.std(axis=1)
          mean = mean / (i + 1)
          std = std / (i + 1)
          
          transform_dict = {'mean': mean[:, np.newaxis], 'std': std[:, np.newaxis],
                            'ninput_channels': len(mean)}
          with open(mean_std_cache, 'wb') as f:
              pickle.dump(transform_dict, f)
          print('saved: ', mean_std_cache)
          self.num_aug = num_aug
          
      # open mean / std from file
      with open(mean_std_cache, 'rb') as f:
          transform_dict = pickle.load(f)
          print('loaded mean / std from cache')
          self.mean = transform_dict['mean']
          self.std = transform_dict['std']
          self.ninput_channels = transform_dict['ninput_channels']

    # this is when the folders are organized by class...
    @staticmethod
    def find_classes(dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    @staticmethod
    def make_dataset_by_class(dir, class_to_idx, phase):
        meshes = []
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if is_mesh_file(fname) and (root.count(phase)==1):
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        meshes.append(item)
        return meshes


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.array([d[key] for d in batch])})
    return meta