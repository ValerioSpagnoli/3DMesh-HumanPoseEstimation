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
                 ninput_edges=750, 
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

        self.device = device
        self.is_train = is_train
        
        net = networks.MeshClassifier(ninput_channels, nclasses, ninput_edges)
        self.net = networks.init_weights(net, init_type, init_gain)
        self.net.to(device)
        
        self.net.train(is_train)
        self.criterion = torch.nn.CrossEntropyLoss().to(device)

        if is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, betas=(beta1, 0.999))
            print_network(self.net)
        else:
            self.load_network(load_epoch)
            

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
        labels = self.labels.repeat(out.shape[0])
        self.loss = self.criterion(out, labels)
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