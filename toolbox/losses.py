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

class Perm_Penal(nn.Module):
    def __init__(self,coeff = 1):
        super(Perm_Penal,self).__init__()
        self.coeff = coeff
    
    def forward(self,m):
        """
        Matrix of size (bs,n_vertices,n_vertices)
        """
        n_vertices = m.shape[-1]
        m_abs = torch.abs(m)
        loss_i = torch.sum( (m_abs.sum(keepdims=True,dim=-1) - torch.sqrt( (m**2).sum(keepdims=True,dim=-1) )) )
        loss_j = torch.sum( (m_abs.sum(keepdims=True,dim=-2) - torch.sqrt( (m**2).sum(keepdims=True,dim=-2) )) )

        MSE = torch.nn.MSELoss(reduction='sum')
        loss_sum_i = MSE(torch.sum(m,dim=-1,keepdims=True),torch.ones((1,n_vertices,1)))
        loss_sum_j = MSE(torch.sum(m,dim=-2,keepdims=True),torch.ones((1,1,n_vertices)))

        total_loss = loss_i + loss_j + self.coeff*( loss_sum_i + loss_sum_j )

        return total_loss


class tsp_perm_loss(nn.Module):
    def __init__(self, loss=nn.BCELoss(reduction='none'), lambda_perm = 1e-2, perm_coeff = 1):
        super(tsp_perm_loss, self).__init__()
        self.base_loss = loss
        self.normalize = torch.nn.Sigmoid()#Softmax(dim=2)
        self.lambda_perm = lambda_perm
        self.perm_penality = Perm_Penal(coeff = perm_coeff)
        
    def forward(self, raw_scores, target):#Used to have also a mask as argument -> Ask MLelarge
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        perm_contrib = self.perm_penality(raw_scores)

        proba = self.normalize(raw_scores)
        loss = self.base_loss(proba,target)
        return torch.mean(loss) + self.lambda_perm * perm_contrib # Was return torch.mean(mask*self.loss(proba,target))

class tsp_fiedler_loss(nn.Module):
    def __init__(self, fiedler_coeff = 1e-2, n_vertices=None, loss=nn.BCELoss(reduction='none')):
        super(tsp_fiedler_loss, self).__init__()
        self.base_loss = loss
        self.normalize = torch.nn.Sigmoid()#Softmax(dim=2)
        self.fiedler_coeff = fiedler_coeff
        self.n_vertices = -1
        self.fiedler_opti = 0
        if n_vertices is not None:
            self._fiedler_init(n_vertices)
    
    def _fiedler_compute(self,n_vertices):
        if n_vertices!=self.n_vertices:
            assert n_vertices>=3, "n_vertices smaller than 3 !"
            self.n_vertices = n_vertices
            M = 2*torch.eye(self.n_vertices)
            temp_id = -1*torch.eye(self.n_vertices)
            M += torch.roll(temp_id,1,dims=-2)
            M += torch.roll(temp_id,-1,dims=-2)
            eigvals,_ = torch.symeig(M)
            self.fiedler_opti = eigvals[1]
            print(f"Fiedler value changed to {self.fiedler_opti}")

        
    def forward(self, raw_scores, target):
        """
        raw_scores (bs,n_vertices,n_vertices)
        """
        n_vertices = raw_scores.shape[1]
        self._fiedler_compute(n_vertices)

        proba = self.normalize(raw_scores)
        base_loss = self.base_loss(proba,target)

        device = 'cpu'
        if raw_scores.is_cuda:
            device = raw_scores.get_device()
        
        _,ind = torch.topk(raw_scores,k=2,dim=2)
        y_onehot = torch.zeros_like(raw_scores).to(device)
        y_onehot.scatter_(2, ind, 1)
        y_onehot = torch.sign(y_onehot * y_onehot.transpose(-2,-1)) #Symmetrization of the matrix
        degrees = y_onehot.sum(axis=2)
        degrees = torch.diag_embed(degrees)
        lap = degrees - y_onehot
        eigvals, _ = torch.symeig(lap,eigenvectors=True)
        fiedler_val = eigvals[:,1]
        
        base_loss_flattened = torch.mean(torch.mean(base_loss,dim=-1),dim=-1) #base_loss_flattened.shape = batch_size

        fiedler_loss_fn = torch.nn.MSELoss()
        fiedler_loss = fiedler_loss_fn(fiedler_val,self.fiedler_opti)

        return torch.mean(base_loss_flattened + self.fiedler_coeff * fiedler_loss) # Was return torch.mean(mask*self.loss(proba,target))
 