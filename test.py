import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test(KSTEPS=20, D_COL=0.2):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    col_avg_ls = []
    col_minade_ls = []
    raw_data_dict = {}
    step =0
    for batch in loader_test: 
        step+=1
        #Get data
        batch = [tensor.to(device) for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr = batch


        num_of_objs = obs_traj_rel.shape[1]

        #Forward
        #V_obs = batch,seq,node,feat
        #V_obs_tmp = batch,feat,seq,node
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze())
        # print(V_pred.shape)
        # torch.Size([1, 5, 12, 2])
        # torch.Size([12, 2, 5])
        V_pred = V_pred.permute(0,2,3,1)
        # torch.Size([1, 12, 2, 5])>>seq,node,feat
        # V_pred= torch.rand_like(V_tr).cuda()


        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        #print(V_pred.shape)

        #For now I have my bi-variate parameters 
        #normx =  V_pred[:,:,0:1]
        #normy =  V_pred[:,:,1:2]
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2).to(device)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
            
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))
        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

        # Scene-level collision metrics over the KSTEPS samples.
        # Only peds with valid GT (:num_of_objs) are considered.
        samples_scene = [s[:, :num_of_objs, :] for s in raw_data_dict[step]['pred']]
        target_scene = V_y_rel_to_abs[:, :num_of_objs, :]
        col_res = collision_metrics(samples_scene, target_scene, d_col=D_COL)
        col_avg_ls.append(col_res['ColRate_avg'])
        col_minade_ls.append(col_res['ColRate_minADE'])

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    col_avg_ = sum(col_avg_ls)/len(col_avg_ls)
    col_minade_ = sum(col_minade_ls)/len(col_minade_ls)
    return ade_,fde_,col_avg_,col_minade_,raw_data_dict


if __name__ == '__main__':
    paths = ['./checkpoint/*social-stgcnn*']
    KSTEPS=20

    print("*"*50)
    print('Number of samples:',KSTEPS)
    print("*"*50)




    for feta in range(len(paths)):
        ade_ls = []
        fde_ls = []
        col_avg_ls = []
        col_minade_ls = []
        path = paths[feta]
        exps = glob.glob(path)
        print('Model being tested are:',exps)

        for exp_path in exps:
            print("*"*50)
            print("Evaluating model:",exp_path)

            model_path = exp_path+'/val_best.pth'
            args_path = exp_path+'/args.pkl'
            with open(args_path,'rb') as f:
                args = pickle.load(f)

            stats= exp_path+'/constant_metrics.pkl'
            with open(stats,'rb') as f:
                cm = pickle.load(f)
            print("Stats:",cm)



            #Data prep
            obs_seq_len = args.obs_seq_len
            pred_seq_len = args.pred_seq_len
            data_set = './datasets/'+args.dataset+'/'

            dset_test = TrajectoryDataset(
                    data_set+'test/',
                    obs_len=obs_seq_len,
                    pred_len=pred_seq_len,
                    skip=1,norm_lap_matr=True)

            loader_test = DataLoader(
                    dset_test,
                    batch_size=1,#This is irrelative to the args batch size parameter
                    shuffle =False,
                    num_workers=0)



            #Defining the model
            model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
            output_feat=args.output_size,seq_len=args.obs_seq_len,
            kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))


            ade_ =999999
            fde_ =999999
            print("Testing ....")
            ad,fd,col_avg,col_minade,raw_data_dic_= test()
            ade_= min(ade_,ad)
            fde_ =min(fde_,fd)
            ade_ls.append(ade_)
            fde_ls.append(fde_)
            col_avg_ls.append(col_avg)
            col_minade_ls.append(col_minade)
            print("ADE:",ade_," FDE:",fde_,
                  " ColRate_avg:",round(col_avg,4),
                  " ColRate_minADE:",round(col_minade,4))




        print("*"*50)

        n = len(ade_ls) if ade_ls else 1
        print("Avg ADE:",sum(ade_ls)/n)
        print("Avg FDE:",sum(fde_ls)/n)
        print("Avg ColRate_avg:",sum(col_avg_ls)/n)
        print("Avg ColRate_minADE:",sum(col_minade_ls)/n)