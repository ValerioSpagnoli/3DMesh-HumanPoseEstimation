import torch
from . import networks
import os
from utils.util import print_network

class ClassifierModel:
    def __init__(self, 
                 is_train, 
                 device, 
                 nclasses, 
                 ninput_channels, 
                 checkpoints_dir='checkpoints', 
                 ncf=[64, 128, 256, 256], 
                 ninput_edges=750, 
                 pool_res=[600, 450, 300, 180],
                 fc_n=100, 
                 resblocks=3, 
                 norm_type='batch', 
                 num_groups=16, 
                 init_type='normal', 
                 init_gain=0.02, 
                 lr=0.0001, 
                 beta1=0.9, 
                 load_epoch=1):
        
        self.save_dir = os.path.join(checkpoints_dir)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None

        norm_layer = networks.get_norm_layer(norm_type=norm_type, num_groups=num_groups)
        net = networks.MeshConvNet(norm_layer, ninput_channels, ncf, nclasses, ninput_edges, pool_res, fc_n, resblocks)
        self.net = networks.init_weights(net, init_type, init_gain)

        self.net.train(is_train)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

        if is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(beta1, 0.999))
            print_network(self.net)
        else:
            self.load_network(load_epoch)
            
        self.device = device
        self.is_train = is_train

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        labels = torch.from_numpy(data['label']).long()
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']

    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out
    
    def backward(self, out):
        self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()

    def load_network(self, load_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % load_epoch
        load_path = os.path.join(self.save_dir, save_filename)
        net = self.net
        print('loading the model from %s' % load_path)
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'): del state_dict._metadata
        net.load_state_dict(state_dict)

    def save_network(self, load_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (load_epoch)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(self.net.cpu().state_dict(), save_path)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            pred_class = out.data.max(1)[1]
            label_class = self.labels
            correct = self.get_accuracy(pred_class, label_class)
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification """
        correct = pred.eq(labels).sum()
        return correct