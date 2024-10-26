import os
import yaml
from collections import defaultdict
from munch import Munch

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from sklearn.preprocessing import StandardScaler,PowerTransformer,MinMaxScaler,QuantileTransformer
from sklearn.preprocessing import PowerTransformer

from scipy.stats import gmean
import numpy as np
from tqdm import tqdm

import models
import matplotlib.pyplot as plt


def predicting_and_testing_sklearn(model,feature,targets,y_train,y_scaler):
    predictions = model.predict(feature)
    if(len(predictions.shape) == 1):
        predictions = np.expand_dims(predictions,axis=-1)
        
    outputs = torch.from_numpy(y_scaler.inverse_transform((predictions).astype('float32'))).float()
    
    task_metrics = testing_shots_regions(outputs.squeeze(1),targets.squeeze(1),torch.from_numpy(y_scaler.inverse_transform(y_train).astype('float32')).float().squeeze(1))
    return [task_metrics['overall']['rmse'],task_metrics['many']['rmse'],task_metrics['medium']['rmse'],task_metrics['few']['rmse']]

def predicting_and_testing_sklearn_2(model,feature,targets,y_train):
    predictions = model.predict(feature)
    if(len(predictions.shape) == 1):
        predictions = np.expand_dims(predictions,axis=-1)

    return predictions
    
class BatchingTestSample(Dataset):
    def __init__(self, x_train,x_test,y_train,y_test,indices,dim_model,in_con_dim,in_con_size):
        
        self.x_train = x_train.unsqueeze(0)
        self.x_test = x_test
        self.y_train = y_train.unsqueeze(0)
        self.y_test = y_test

        self.indices = indices
        
        self.feature_size = x_train.shape[-1]
        self.dim_model = dim_model
        self.in_con_size = in_con_size
        self.in_con_dim = in_con_dim

        #In the case of large feature, feed all of them in the same batch 
        if(self.feature_size<=in_con_dim):
            self.correct_dim = in_con_dim
        else:
            self.correct_dim = (self.feature_size//in_con_dim+1)*in_con_dim
            
    def __len__(self):
        return len(self.x_test)

    def __getitem__(self, idx):
        x_context = torch.cat((self.x_train[:,self.indices[idx],:],self.x_test[idx,:].unsqueeze(0).unsqueeze(0)),1)
        
        #get the x_context in the correct dimension to by divided in chunk 
        batched_x_context = torch.zeros(1,self.in_con_size + 1 ,self.correct_dim)
        batched_x_context[:,:,:self.feature_size] = x_context

        #devide in chunk
        batched_x_context = batched_x_context.permute(0,2,1).reshape(-1,self.in_con_dim,self.in_con_size + 1).permute(0,2,1)

        #adjust the chunk dimension to input model 
        plac_hold = torch.zeros((batched_x_context.shape[0],batched_x_context.shape[1],self.dim_model))
        plac_hold[:,:,:self.in_con_dim] = batched_x_context
        batched_x_context = plac_hold
        
        #do the same for the y 
        batched_y_context = torch.cat((self.y_train[:,self.indices[idx],:],torch.zeros((1,1,1))),1).repeat(batched_x_context.shape[0],1,1)
        return batched_x_context,batched_y_context

class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def testing_in_contect_pfn(model,dataloader,in_con_size,n_dims,device):
    predictions = []
    for i, (x,y) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch_size, chunk, _, _ = x.shape 
            x = x.view(-1,in_con_size+1,n_dims).float().to(device).permute(1,0,2)
            y = y.view(-1,in_con_size+1,1).float().to(device).permute(1,0,2)
            pred = model((None,x,y),single_eval_pos = in_con_size)#[:,-1] #get only the last for the test sample
            pred = model.criterion.mean(pred.softmax(-1).log_())
            pred = pred.view(batch_size,chunk,)
            pred = pred.mean(1).detach().cpu().numpy()
            predictions.append(pred)
    predictions = np.hstack(predictions)
    return predictions
    
    

def testing_in_contect(model,dataloader,in_con_size,n_dims,device):
    predictions = []
    for i, (x,y) in enumerate(tqdm(dataloader)):
        with torch.no_grad():
            batch_size, chunk, _, _ = x.shape 
            x = x.view(-1,in_con_size+1,n_dims).float().to(device)
            y = y.view(-1,in_con_size+1,1).float().to(device)
            pred = model(x,y)[:,-1] #get only the last for the test sample
        
            pred = pred.view(batch_size,chunk)
            pred = pred.mean(1).detach().cpu().numpy()
            predictions.append(pred)
    
    predictions = np.hstack(predictions)
    return predictions
    
def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = models.build_model(conf.model)

    if step == -1:
        state_path = os.path.join(run_path, "state.pt")
        state = torch.load(state_path,map_location=torch.device('cpu'))
        model.load_state_dict(state["model_state_dict"])
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        state_dict = torch.load(model_path,map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

    return model, conf

def normalize_data(data_train,data_test,scaler_type = 'std'):
    if(scaler_type == 'power_transform'):
        scaler = PowerTransformer(method="yeo-johnson")
    elif(scaler_type == 'min_max'):
        scaler = MinMaxScaler()
    elif(scaler_type == 'qt'):
        scaler = QuantileTransformer(n_quantiles=50, random_state=0)
    else:
        scaler = StandardScaler()
        
    data_train = scaler.fit_transform(data_train)
    data_test = scaler.transform(data_test)
    return torch.from_numpy(data_train),torch.from_numpy(data_test),scaler

def testing_loop(outputs,targets,train_labels):
    batch_time = AverageMeter('Time', ':6.3f')
    losses_mse = AverageMeter('Loss (MSE)', ':.3f')
    losses_l1 = AverageMeter('Loss (L1)', ':.3f')
    
    criterion_mse = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_gmean = nn.L1Loss(reduction='none')
    
    losses_all = []
    
    loss_mse = criterion_mse(outputs, targets)
    loss_l1 = criterion_l1(outputs, targets)
    loss_all = criterion_gmean(outputs, targets)
    losses_all.extend(loss_all.cpu().numpy())
    losses_mse.update(loss_mse.item(), outputs.size(0))
    losses_l1.update(loss_l1.item(), outputs.size(0))
    
    shot_dict = shot_metrics(outputs, targets, train_labels)
    loss_gmean = gmean(np.hstack(losses_all), axis=None).astype(float)
    
    print(f" * Overall: MSE {losses_mse.avg:.3f}\tL1 {losses_l1.avg:.3f}\tG-Mean {loss_gmean:.3f}")
    print(f" * Many: MSE {shot_dict['many']['mse']:.3f}\t"
          f"L1 {shot_dict['many']['l1']:.3f}\tG-Mean {shot_dict['many']['gmean']:.3f}")
    print(f" * Median: MSE {shot_dict['median']['mse']:.3f}\t"
          f"L1 {shot_dict['median']['l1']:.3f}\tG-Mean {shot_dict['median']['gmean']:.3f}")
    print(f" * Low: MSE {shot_dict['low']['mse']:.3f}\t"
          f"L1 {shot_dict['low']['l1']:.3f}\tG-Mean {shot_dict['low']['gmean']:.3f}")
    return shot_dict

def shot_metrics(preds, labels, train_labels, many_shot_thr=100, low_shot_thr=20):
    train_labels = np.array(train_labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError(f'Type ({type(preds)}) of predictions not supported')

    train_class_count, test_class_count = [], []
    mse_per_class, l1_per_class, l1_all_per_class = [], [], []
    for l in np.unique(labels):
        train_class_count.append(len(train_labels[train_labels == l]))
        test_class_count.append(len(labels[labels == l]))
        mse_per_class.append(np.sum((preds[labels == l] - labels[labels == l]) ** 2))
        l1_per_class.append(np.sum(np.abs(preds[labels == l] - labels[labels == l])))
        l1_all_per_class.append(np.abs(preds[labels == l] - labels[labels == l]))

    many_shot_mse, median_shot_mse, low_shot_mse = [], [], []
    many_shot_l1, median_shot_l1, low_shot_l1 = [], [], []
    many_shot_gmean, median_shot_gmean, low_shot_gmean = [], [], []
    many_shot_cnt, median_shot_cnt, low_shot_cnt = [], [], []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot_mse.append(mse_per_class[i])
            many_shot_l1.append(l1_per_class[i])
            many_shot_gmean += list(l1_all_per_class[i])
            many_shot_cnt.append(test_class_count[i])
        elif train_class_count[i] < low_shot_thr:
            low_shot_mse.append(mse_per_class[i])
            low_shot_l1.append(l1_per_class[i])
            low_shot_gmean += list(l1_all_per_class[i])
            low_shot_cnt.append(test_class_count[i])
        else:
            median_shot_mse.append(mse_per_class[i])
            median_shot_l1.append(l1_per_class[i])
            median_shot_gmean += list(l1_all_per_class[i])
            median_shot_cnt.append(test_class_count[i])

    shot_dict = defaultdict(dict)
    shot_dict['many']['mse'] = np.sum(many_shot_mse) / np.sum(many_shot_cnt)
    shot_dict['many']['l1'] = np.sum(many_shot_l1) / np.sum(many_shot_cnt)
    shot_dict['many']['gmean'] = gmean(np.hstack(many_shot_gmean), axis=None).astype(float)
    shot_dict['median']['mse'] = np.sum(median_shot_mse) / np.sum(median_shot_cnt)
    shot_dict['median']['l1'] = np.sum(median_shot_l1) / np.sum(median_shot_cnt)
    shot_dict['median']['gmean'] = gmean(np.hstack(median_shot_gmean), axis=None).astype(float)
    shot_dict['low']['mse'] = np.sum(low_shot_mse) / np.sum(low_shot_cnt)
    shot_dict['low']['l1'] = np.sum(low_shot_l1) / np.sum(low_shot_cnt)
    shot_dict['low']['gmean'] = gmean(np.hstack(low_shot_gmean), axis=None).astype(float)

    return shot_dict

def get_bin_idx(label):
    _, bins_edges = np.histogram(a=np.array([], dtype=np.float32), bins=50, range=(0., 5.))
    if label == 5.:
        return 50 - 1
    else:
        return np.where(bins_edges > label)[0][0] - 1
    
def evaluate_model_sts(pred,label):
    num_bins = 50
    shot_idx = {
                'many': [0, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 49],
                'medium': [2, 4, 6, 8, 27, 35, 37],
                'few': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 29, 31, 33, 39, 41, 43, 45, 47]
            }
    
    count = len(label)

    label_bin_idx = list(map(get_bin_idx, label))

    def bin2shot(idx):
        if idx in shot_idx['many']:
            return 'many'
        elif idx in shot_idx['medium']:
            return 'medium'
        else:
            return 'few'

    label_category = np.array(list(map(bin2shot, label_bin_idx)))
    pred_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
    label_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
    metric = {'many': {}, 'medium': {}, 'few': {}, 'overall': {}}

    for shot in ['overall', 'many', 'medium', 'few']:
        pred_shot[shot] = np.array(pred)[label_category == shot] if shot != 'overall' else np.array(pred)
        label_shot[shot] = np.array(label)[label_category == shot] if shot != 'overall' else np.array(label)
        metric[shot]['mse'] = np.mean((pred_shot[shot] - label_shot[shot]) ** 2) if pred_shot[shot].size > 0 else 0.
        if pred_shot[shot].size <= 0:
            metric[shot]['gmean'] = 0.
        else:
            diff = np.abs(pred_shot[shot] - label_shot[shot])
            if diff[diff == 0.].size:
                diff[diff == 0.] += 1e-10
                metric[shot]['gmean'] = gmean(diff) if pred_shot[shot].size > 0 else 0.
            else:
                metric[shot]['gmean'] = gmean(np.abs(pred_shot[shot] - label_shot[shot])) if pred_shot[shot].size > 0 else 0.
        metric[shot]['num_samples'] = pred_shot[shot].size
    task_metrics = metric

    print(f" * Overall: MSE {task_metrics['overall']['mse']:.3f}\tG-Mean {task_metrics['overall']['gmean'][0]:.3f}")
    print(f" * Many: MSE {task_metrics['many']['mse']:.3f}\t"
          f"G-Mean {task_metrics['many']['gmean'][0]:.3f}")
    print(f" * Median: MSE {task_metrics['medium']['mse']:.3f}\t"
          f"G-Mean {task_metrics['medium']['gmean'][0]:.3f}")
    print(f" * Low: MSE {task_metrics['few']['mse']:.3f}\t"
          f"G-Mean {task_metrics['few']['gmean'][0]:.3f}")
    return task_metrics['overall']['mse']


def discretize_distribution(distribution, bins):
    # Convert continuous distribution to discrete by fixed bins
    counts, bin_edges = np.histogram(distribution, bins=bins)
    discrete_distribution = (bin_edges[:-1] + bin_edges[1:]) / 2
    return discrete_distribution, counts, bin_edges

def classify_bin_counts(counts):
    # Classify each bin's count independently
    return ["few" if count < 20 else "medium" if 20 <= count <= 100 else "many" for count in counts]

def create_grouped_region_mapping(classifications):
    # Create a dictionary grouping indices by their classifications
    mapping = {"few": [], "medium": [], "many": []}
    for index, region in enumerate(classifications):
        mapping[region].append(index)
    return mapping

def map_test_samples_to_regions(test_samples, bin_edges, region_mapping,bin_max):
    # Determine which bin each test sample falls into
    bin_indices = np.digitize(test_samples, bin_edges, right=False) - 1 
    regions = []
    for bin_index in bin_indices:
        if bin_index < 0 or bin_index >= bin_max:
            # Directly assign out-of-range values to the 'few' category.
            regions.append('few')
        else:
            # Find the appropriate region for in-range values.
            region = next((key for key, indices in region_mapping.items() if bin_index in indices), 'few')
            regions.append(region)
    return regions

def testing_shots_regions(pred,label,training_distribution):
    if(training_distribution.shape[0]<= 1000):
        Bins = 10
    else:
        Bins = 50
    
    if(len(list(set(training_distribution.numpy())))<Bins):
        Bins = len(list(set(training_distribution.numpy())))
        
    # Discretize the training distribution
    discrete_training, counts_training, bin_edges = discretize_distribution(training_distribution, bins=Bins)
    # Classify each bin based on its count
    classification = classify_bin_counts(counts_training)
    # Create a mapping from index to region
    region_mapping  = create_grouped_region_mapping(classification)
    
    # Map test samples to regions
    label_category = np.array(map_test_samples_to_regions(label, bin_edges, region_mapping,bin_max = Bins))
    
    pred_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
    label_shot = {'many': [], 'medium': [], 'few': [], 'overall': []}
    metric = {'many': {}, 'medium': {}, 'few': {}, 'overall': {}}
    
    for shot in ['overall', 'many', 'medium', 'few']:
        pred_shot[shot] = np.array(pred)[label_category == shot] if shot != 'overall' else np.array(pred)
        label_shot[shot] = np.array(label)[label_category == shot] if shot != 'overall' else np.array(label)
        metric[shot]['rmse'] = np.sqrt(np.mean((pred_shot[shot] - label_shot[shot]) ** 2)) if pred_shot[shot].size > 0 else 0.
        if pred_shot[shot].size <= 0:
            metric[shot]['gmean'] = 0.
        else:
            diff = np.abs(pred_shot[shot] - label_shot[shot])
            if diff[diff == 0.].size:
                diff[diff == 0.] += 1e-10
                metric[shot]['gmean'] = gmean(diff) if pred_shot[shot].size > 0 else 0.
            else:
                metric[shot]['gmean'] = gmean(np.abs(pred_shot[shot] - label_shot[shot])) if pred_shot[shot].size > 0 else 0.
        metric[shot]['num_samples'] = pred_shot[shot].size
    task_metrics = metric
    
    print(f" * Overall: RMSE {task_metrics['overall']['rmse']:.3f}\tG-Mean {task_metrics['overall']['gmean']:.3f}")
    print(f" * Many: RMSE {task_metrics['many']['rmse']:.3f}\t"
          f"G-Mean {task_metrics['many']['gmean']:.3f}")
    print(f" * Median: RMSE {task_metrics['medium']['rmse']:.3f}\t"
          f"G-Mean {task_metrics['medium']['gmean']:.3f}")
    print(f" * Low: RMSE {task_metrics['few']['rmse']:.3f}\t"
          f"G-Mean {task_metrics['few']['gmean']:.3f}")

    return task_metrics
