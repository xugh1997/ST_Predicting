import yaml,torch,os, gc
import numpy as np
import torch.nn as nn
from dataloader import data_generator
from ST_BLOCK import SerialBlock
from tqdm import tqdm

import torch.optim as optim
import random, time
import sys, copy
from datetime import datetime
# from plot import plot_scatter_x_y_pre_true, plot_scatter_x_y_two_tensor, plot_curve
from compute_loss import compute_mae, compute_mape, R_Square


def schedule_lr(patience, epoch, save_dict):
    if (((epoch-save_dict['epoch'])%patience)==0) and (epoch!=save_dict['epoch']):
        return True
    else:
        return False
def load_config(yml_path):
    with open(yml_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# load dataset#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def initialize_xavier(model):
    """
    使用 Xavier 初始化模型的所有参数。
    Args:
        model (nn.Module): 需要初始化的 PyTorch 模型。
    """    

    for name, p in model.named_parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    total_params = sum(p.numel() for p in model.parameters())
    for name, param in model.named_parameters():
        print(f'当前的参数为{name},参数量为{param.numel()}')

    print(f"Total parameters: {total_params}")
def save_model(path, **save_dict):
    os.makedirs(os.path.split(path)[0], exist_ok=True)
    torch.save(save_dict,path)


def test(test_loader):
    total_loss_mse = 0
    total_loss_mae = 0
    total_loss_mape = 0
    total_loss_rmse = 0
    total_loss_rmse_grid = 0
    total_loss_rsquare = 0
    # save_path = os.path.join(model_parameter_folder,'best_model_without_constrain_hygcn.pkl')
    save_path = os.path.join('model_save_path','best_model(data0.2_hyper_edge0.2_without_init).pkl')
    save_dict = torch.load(save_path, weights_only=False)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    Y_plot_true=list()
    Y_plot_pre = list()
    one_batch = []
    with torch.no_grad():
        for X,Y in tqdm(test_loader):
            y_predict = model(X.unsqueeze(-1).to(device))
            Y = Y.permute(0, 2, 1).to(device)
            Y_plot_true.append(Y)
            Y_plot_pre.append(y_predict)
            rmse_loss = torch.sqrt(torch.mean((Y - y_predict) ** 2))*len(X)
            # rmse_loss_grid = torch.sqrt(torch.mean((Y - y_predict) ** 2,dim=0))*len(X)
            mse_loss = torch.nn.functional.mse_loss(y_predict,Y)*len(X)
            mae_loss = compute_mae(y_predict, Y)*len(X)
            mape_loss = compute_mape(y_predict, Y)*len(X)
            r2_loss = R_Square(Y.squeeze(), y_predict.squeeze())*len(X)
            total_loss_rmse += rmse_loss.item()
            # total_loss_rmse_grid += rmse_loss_grid.cpu().detach().numpy()
            total_loss_mae += mae_loss.item()
            total_loss_mape += mape_loss.item()
            total_loss_mse += mse_loss.item()
            total_loss_rsquare += r2_loss.item()
            one_batch.append(y_predict)
            torch.cuda.empty_cache()
        Y_plot_pre = torch.cat(Y_plot_pre,dim=0)
        Y_plot_true = torch.cat(Y_plot_true, dim=0)
    # plot_scatter_x_y_pre_true(Y_plot_pre, Y_plot_true, 'TEST')
    # plot_scatter_x_y_two_tensor(Y_plot_pre, Y_plot_true, 'TEST')
    res_str = f'当前时间{datetime.now()}最佳模型的训练误差RMSE为{total_loss_rmse/len(test_loader.dataset)},MAE误差为{total_loss_mae/len(test_loader.dataset)},MAPE误差为{total_loss_mape/len(test_loader.dataset)},MSE误差为{total_loss_mse/len(test_loader.dataset)},R2为{total_loss_rsquare/len(test_loader.dataset)}'
    # print(res_str)
    total_loss_rmse_grid_pt = total_loss_rmse_grid/len(test_loader.dataset)
    os.makedirs('./result_error',exist_ok=True)
    result = torch.concat(one_batch,dim=0)
    rmse_whole = torch.sqrt(torch.mean((Y_plot_pre-Y_plot_true)**2))
    mae_whole = torch.mean(torch.abs(Y_plot_pre-Y_plot_true))
    R_whole = R_Square(Y_plot_true, Y_plot_pre)
    mape_whole = compute_mape(Y_plot_true, Y_plot_pre)
    res_str2 =  f'当前时间{datetime.now()},最佳模型的训练误差RMSE为{rmse_whole},MAE误差为{mae_whole},R2为{R_whole}, MAPE为{mape_whole}_data0.2_hyper_edge0.2_without_init'
    print(res_str2)
    with open(os.path.join('./result_error','result.txt'),'a+') as f:
      f.write(res_str2+'\n')


def main(model):
    epoch_num = configs['epochs']
    model_parameter_folder = 'model_save_path'
    save_path = os.path.join(model_parameter_folder,'best_modeltest.pkl')
 
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_loader, valid_loader, test_loader = data_generator('data/after_processed/previous_version', configs)
    if os.path.exists(save_path):
        print('path exists')
        save_dict = torch.load(save_path,weights_only=False)
        model.load_state_dict(save_dict['model_state_dict'])

        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch']
        duration_ = save_dict['duration']
       
        # running_loss_record['validate_mae_std'] = copy.deepcopy(save_dict['valid_curve']['validate_mae_std'])
        # running_loss_rmse = dict()
        # running_loss_record['validate_total_loss']=copy.deepcopy(save_dict['valid_curve']['validte_total_loss'])
        #move the load parameter tensor to cuda, for optimizer.step()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
    else:
        print('path not exists')
        save_dict = dict()
        duration_ = 0.0
        best_val_loss = float('inf')
        begin_epoch  = 0
        
        # running_loss_record['validate_total_loss'] =[]

    flag=0
    phases = ['train', 'validate']
    start_time = time.time()
    for epoch in range(begin_epoch,epoch_num):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'当前为{epoch+1}轮训练和测试,学习率为{current_lr}')
        for ph in phases:
            total_loss_mse = 0
            total_loss_mape = 0
            total_loss_rmse = 0
            total_loss_mae = 0
            total_loss_rmse_std = 0
            total_loss_mae_std = 0
            if ph == 'train':
                model.train()
                for X,Y in tqdm(train_loader):
                    y_predict = model(X.unsqueeze(-1).to(device))
                    Y = Y.permute(0,2,1).to(device)
                  
                    loss_rmse = torch.sqrt(torch.mean((Y - y_predict) ** 2))
                    total_loss_rmse += ((loss_rmse.cpu().detach().numpy())*len(X))
                    loss_mse = criterion(y_predict,Y)
                    total_loss_mse += loss_mse.item()*len(X)
                    loss_mape = compute_mape(Y, y_predict).cpu().detach().numpy()
                    total_loss_mape += loss_mape*len(X)
                    loss_mae = compute_mae(Y,y_predict).cpu().detach().numpy()
                    total_loss_mae += loss_mae*len(X)
                    optimizer.zero_grad()  # 清零梯度
                    loss_rmse.backward()  # 反向传播
                    optimizer.step()
                    del loss_rmse
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                print(f'当前训练误差RMSE为{total_loss_rmse/len(train_loader.dataset)}MSE为{total_loss_mse/len(train_loader.dataset)}MAPE为{total_loss_mape/len(train_loader.dataset)}MAE为{total_loss_mae/len(train_loader.dataset)},学习率为{current_lr}')
                
            else:
                model.eval()
                with torch.no_grad():
                    for X,Y in tqdm(valid_loader):
                        y_predict = model(X.unsqueeze(-1).to(device))
                        Y = Y.permute(0,2,1).to(device)
                        loss_mse = torch.nn.functional.mse_loss(y_predict, Y).cpu().detach().numpy()
                        total_loss_mse += loss_mse * len(X)
                        loss_mape = compute_mape(Y, y_predict).cpu().detach().numpy()
                        total_loss_mape += loss_mape * len(X)
                        loss_rmse = np.sqrt(loss_mse)
                        total_loss_rmse += loss_rmse*len(X)
                        loss_mae = compute_mae(Y,y_predict).cpu().detach().numpy()
                        total_loss_mae += loss_mae*len(X)
                        rmse_std = torch.sqrt(torch.std((y_predict-Y)**2)).item()
                        total_loss_rmse_std += rmse_std*len(X)
                        mae_std = torch.std(torch.abs(y_predict-Y)).item()
                        total_loss_mae_std += mae_std*len(X)
                        torch.cuda.empty_cache()
                    print(f'当前测试误差RMSE为{total_loss_rmse/len(valid_loader.dataset)},MSE为{total_loss_mse/len(valid_loader.dataset)},MAPE为{total_loss_mape/len(valid_loader.dataset)}MAE为{total_loss_mae/len(valid_loader.dataset)}')
     
            if ph == 'validate':
                if total_loss_rmse/len(valid_loader.dataset) < best_val_loss:
                    print(f"当前的测试误差为{total_loss_rmse/len(valid_loader.dataset)},最优的误差为{best_val_loss},判断当前的误差是否小于最佳误差{total_loss_rmse/len(valid_loader.dataset) <best_val_loss}")
                    best_val_loss = total_loss_rmse/len(valid_loader.dataset) 
                    duration_now = duration_+(time.time()-start_time)
                    save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                    epoch=epoch,
                                    duration=duration_now,
                                    # duration2 = (time.time()-start_time2),
                                    best_val_loss=best_val_loss,
                                    optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                    save_model(save_path, **save_dict)
                    print(f'Better model at epoch {epoch+1} recorded.')
                elif epoch - save_dict['epoch'] > configs['early_stop_steps'] or current_lr <= 1.0e-5 :
                    print('Early stopped.')
                    flag = 1
                    break
                if schedule_lr(configs['ReduceLROnPlateau_patience'], epoch, save_dict):
                    optimizer.param_groups[0]['lr'] *= 0.2 
        if flag==1:
          
            break

    test(test_loader)
  
if __name__ == '__main__':
    criterion = nn.MSELoss()
    # seed_everything()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   
    configs = load_config('config copy.yml')
    

    params = {'DEVICE': device,
            'in_dims': 1,
            'n_nodes': configs['n_nodes'],
            'hidden_dims':128,
            }
    model = SerialBlock(**params)
    initialize_xavier(model)
    main(model)
   