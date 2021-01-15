from loaders.tsp_data_generator import TSPGenerator
import toolbox.metrics as metrics
from models.base_model import Simple_Edge_Embedding
import torch
import toolbox.vision as vision

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

##Arguments

name="train"
args_tspg = {'generative_model': "GaussNormal",'num_examples_train':1000,'n_vertices':10,'distance_used':'EUC_2D', 'path_dataset':"dataset_tsp"}
args_model = {'num_blocks': 4,
    'original_features_num': 2,
    'in_features': 64,
    'out_features': 1,
    'depth_of_mlp': 3}

n_epoch=1000
batch_size=10
shuffle=True

model = Simple_Edge_Embedding(**args_model)
model.to(device)

criterion = torch.nn.BCELoss(reduction='none')
reduction = torch.mean
normalize = torch.nn.Sigmoid()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor = 0.5,patience=20)#2)#StepLR(optimizer, step_size=args['scheduler_step'], gamma=args['scheduler_decay'])
##

tspg = TSPGenerator(name,args_tspg)
tspg.load_dataset()


train_loader = torch.utils.data.DataLoader(tspg, batch_size=batch_size, shuffle=shuffle,num_workers=0)

for epoch in range(n_epoch):
    for i, (data,target,g) in enumerate(train_loader):
        
        target = target.to(device)
        data = data.to(device)
        
        output = model(data).squeeze(-1)
        
        results = metrics.compute_accuracy_tsp(output,target,device=device)
        _, rec, f1 = metrics.compute_f1(output,target,device,topk=3)
        

        probs = normalize(output)
        
        optimizer.zero_grad()
        target = target.to(torch.float32)
        probs = probs.to(torch.float32)

        loss_not_reduced = criterion(probs,target)

        loss = reduction(loss_not_reduced)
        loss.backward()
        optimizer.step()
    scheduler.step(loss)
    print(f"Epoch {epoch}, lr {optimizer.param_groups[0]['lr']:.8f} => loss : {loss:.6f}, rec : {rec:.4f}, f1 : {f1:.4f}, worked = {torch.count_nonzero(results)}/{batch_size}")

#print(data,output,loss_not_reduced,target,sep="\n")
vision.compare(g[0][0],g[0][1],metrics.get_path(output,topk=2)[0],target[0].detach().cpu())