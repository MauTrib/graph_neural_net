import torch
import torch.nn as nn

class triplet_loss(nn.Module):
    def __init__(self, device='cpu', loss_reduction='mean', loss=nn.CrossEntropyLoss(reduction='sum')):
        super(triplet_loss, self).__init__()
        self.device = device
        self.loss = loss
        if loss_reduction == 'mean':
            self.increments = lambda new_loss, n_vertices : (new_loss, n_vertices)
        elif loss_reduction == 'mean_of_mean':
            self.increments = lambda new_loss, n_vertices : (new_loss/n_vertices, 1)
        else:
            raise ValueError('Unknown loss_reduction parameters {}'.format(loss_reduction))

    def forward(self, outputs):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        loss = 0
        total = 0
        for out in outputs:
            n_vertices = out.shape[0]
            ide = torch.arange(n_vertices)
            target = ide.to(self.device)
            incrs = self.increments(self.loss(out, target), n_vertices)
            loss += incrs[0]
            total += incrs[1]
        return loss/total

def get_criterion(device, loss_reduction):
    return triplet_loss(device, loss_reduction)

# TODO refactor

class tsp_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none')):
        super(tsp_loss, self).__init__()
        self.loss = loss
        self.normalize = torch.nn.Sigmoid()#Softmax(dim=2)
        
    def forward(self, raw_scores, target):#Used to have also a mask as argument -> Ask MLelarge
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        proba = self.normalize(raw_scores)
        loss = self.loss(proba,target)
        return torch.mean(loss) # Was return torch.mean(mask*self.loss(proba,target))

class tsp_fiedler_loss(nn.Module):
    def __init__(self, fiedler_coeff = 1e-2, loss=nn.BCELoss(reduction='none')):
        super(tsp_fiedler_loss, self).__init__()
        self.base_loss = loss
        self.normalize = torch.nn.Sigmoid()#Softmax(dim=2)
        self.fiedler_coeff = fiedler_coeff
        
    def forward(self, raw_scores, target):
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        n_vertices = raw_scores.shape[1]
        proba = self.normalize(raw_scores)
        base_loss = self.base_loss(proba,target)

        device = 'cpu'
        if raw_scores.is_cuda:
            device = raw_scores.get_device()
        
        _,ind = torch.topk(raw_scores,k=2,dim=2)
        y_onehot = torch.zeros_like(raw_scores).to(device)
        y_onehot.scatter_(2, ind, 1)
        temp = torch.sign(raw_scores*y_onehot)
        degrees = temp.sum(axis=2)
        degrees = torch.diag_embed(degrees)
        lap = degrees - temp
        eigvals, _ = torch.symeig(lap,eigenvectors=True)

        return torch.mean(base_loss + self.fiedler_coeff/n_vertices * eigvals[-2]) # Was return torch.mean(mask*self.loss(proba,target))
 