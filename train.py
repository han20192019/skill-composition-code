import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import argparse
import wandb
import os
import yaml
import shutil
from torch_geometric.data import Data, Batch
# other files
from utils import *
from models import *
from torch_geometric.nn import GCNConv, GATConv
from pointnet_extractor import DP3Encoder
from torch_geometric.nn import global_mean_pool



class MLP_Language(nn.Module):
    def __init__(self, language_feat_dim, pc_feat_dim, hidden_dims=[512, 512]):
        super(MLP_Language, self).__init__()
        
        layers = []
        in_dim = language_feat_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, pc_feat_dim))  # Final layer maps to pc_feat_dim

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

def main():
    parser = argparse.ArgumentParser(description='Training script for setting various parameters.')
    parser.add_argument('--config', type=str, default='./configs/train_config.yml', help='Path to the configuration YAML file.')
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    num_epochs = config['num_epochs']
    num_diffusion_iters = config['num_diffusion_iters']
    num_train_demos = config['num_train_demos']
    pred_horizon = config['pred_horizon']
    obs_horizon = config['obs_horizon']
    action_horizon = config['action_horizon']
    lr = config['lr']
    weight_decay = config['weight_decay']
    batch_size = config['batch_size']
    dataset_path_dir = config['dataset_path_dir']
    models_save_dir = config['models_save_dir']
    display_name = config['display_name']
    resize_scale = config["resize_scale"]

    if display_name == "default":
        display_name = None
    if config["wandb"]:
        wandb.init(
            project="real_world_training_09_28",
            config=config,
            name=display_name
        )
    else:
        print("warning: wandb flag set to False")


    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)

    dataset_list = []
    combined_stats = []
    num_datasets = 0
    all_data_stats = {'agent_pos': {'min': np.array([-0.15829659,  0.2738431 , -0.0690495 , -2.3559928 , -0.05070972,
        2.0321393 ,  0.5672193 ,  0.00930391,  0.00960855], dtype=np.float32), 'max': np.array([ 0.24690998,  0.932966  ,  0.07711923, -1.3707049 ,  0.06377019,
        2.8304572 ,  1.1111133 ,  0.04      ,  0.04      ], dtype=np.float32)}, 'action': {'min': np.array([-0.15884228,  0.2707608 , -0.06952421, -2.357214  , -0.05161012,
        2.026965  ,  0.5663745 , -1.        ], dtype=np.float32), 'max': np.array([ 0.24830234,  0.9371018 ,  0.07746021, -1.369698  ,  0.06438983,
        2.8355749 ,  1.117754  ,  1.        ], dtype=np.float32)}}

    for entry in dataset_path_dir:
        full_path = entry

        # create dataset from file
        dataset = TrainDataset(
            dataset_path=full_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            id = num_datasets,
            num_demos = num_train_demos,
            resize_scale = resize_scale,
            pretrained = config["use_pretrained"],
            stats = all_data_stats
        )
        num_datasets += 1
        # save training data statistics (min, max) for each dim
        stats = dataset.stats
        dataset_list.append(dataset)
        combined_stats.append(stats)

    
    # import pdb
    # pdb.set_trace()
    # stats_lst = {}
    # result_stats = {}
    # for aa in combined_stats:
    #     for key, data in aa.items():
    #         if key in stats_lst:
    #             stats_lst[key]['min'].append(aa[key]['min'])
    #             stats_lst[key]['max'].append(aa[key]['max'])
    #         else:
    #             stats_lst[key] = {'min': [aa[key]['min']], 'max': [aa[key]['max']]}
    # for key in stats_lst.keys():
    #     result_stats[key] = {'min': np.min(np.array(stats_lst[key]['min']), axis = 0), 'max': np.max(np.array(stats_lst[key]['max']), axis = 0)}
    # import pdb
    # pdb.set_trace()
    
    
    combined_dataset = ConcatDataset(dataset_list)

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    vision_feature_dim = 512
    language_feature_dim = 512
    lowdim_obs_dim = 9
    obs_dim = vision_feature_dim + lowdim_obs_dim + language_feature_dim 
    action_dim = 8

    nets = nn.ModuleDict({})
    noise_schedulers = {}


    vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets['vision_encoder'] = vision_encoder                                        

    obs_dict = {'agent_pos': (9, ), 'point_cloud': (100, )}
    
    #just start with the simplest 3 layer MLP
    pointcloud_encoder_cfg = {
        "in_channels": 3,
        "out_channels": 64,
        "use_layernorm": False,
        "final_norm": "none",  
        "normal_channel": False
    }
    obs_encoder = DP3Encoder(observation_space=obs_dict,
                            img_crop_shape=None,
                            out_channel=64,
                            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
                            use_pc_color=False
                            )
    nets['point_cloud_vision_encoder'] = obs_encoder
    
    
    class GDPModel(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels, heads=4):
            super(GDPModel, self).__init__()
            self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
            self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

        def forward(self, x, edge_index, edge_attr, batch):
            x = F.elu(self.gat1(x, edge_index, edge_attr=edge_attr))
            x = self.gat2(x, edge_index, edge_attr=edge_attr)
            graph_embedding = global_mean_pool(x, batch)  # Aggregate node features into graph-level representation
            return graph_embedding
        

    nets['graph_convolution'] = GDPModel(in_channels=128, hidden_channels=128, out_channels=512)
    
    invariant = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )
    nets['invariant'] = invariant 
    noise_schedulers["single"] = create_injected_noise(num_diffusion_iters)       
    
    
    
    mlp_language = MLP_Language(language_feat_dim=512, pc_feat_dim=512)
    nets['mlp_language'] = mlp_language
    

    nets = nets.to(device)

    # Exponential Moving Average accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=nets.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parameters are not optimized
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=lr, weight_decay=weight_decay)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=config["num_warmup_steps"],
        num_training_steps=len(dataloader) * 3050
    )
    
    if config["use_pretrained"]:
        for param in nets["vision_encoder"].parameters():
            param.requires_grad = False
    # create new checkpoint
    checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, 0)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    save(ema, nets, checkpoint_dir)
    with tqdm(range(1, num_epochs+1), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            if config['wandb']:
                wandb.log({'epoch': epoch_idx})    
            epoch_loss = list()
            # batch loop
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:

                    if config["wandb"]:
                        wandb.log({'learning_rate:': lr_scheduler.get_last_lr()[0]})
                    

                    npoint_cloud_0 = nbatch['point_cloud_0'][:,:obs_horizon].to(device, dtype = torch.float32)
                    npoint_cloud_1 = nbatch['point_cloud_1'][:,:obs_horizon].to(device, dtype = torch.float32)
                    npoint_cloud_2 = nbatch['point_cloud_2'][:,:obs_horizon].to(device, dtype = torch.float32)
                    
                    relationship = nbatch['relationship'][:,:obs_horizon].to(device, dtype = torch.float32)
                    instruction = nbatch['instruction'][:,:obs_horizon].to(device, dtype = torch.float32)


                    nagent_pos = nbatch['agent_pos'][:,:obs_horizon].to(device, dtype = torch.float32)
                    naction = nbatch['action'].to(device, dtype = torch.float32)
                    B = nagent_pos.shape[0]
        
                    embedded_pc_0 = nets["point_cloud_vision_encoder"]({
                        "point_cloud": npoint_cloud_0.flatten(end_dim=1),
                        "agent_pos": nagent_pos.flatten(end_dim=1)
                    })
                    
                    embedded_pc_1 = nets["point_cloud_vision_encoder"]({
                        "point_cloud": npoint_cloud_1.flatten(end_dim=1),
                        "agent_pos": nagent_pos.flatten(end_dim=1)
                    })
                    
                    embedded_pc_2 = nets["point_cloud_vision_encoder"]({
                        "point_cloud": npoint_cloud_2.flatten(end_dim=1),
                        "agent_pos": nagent_pos.flatten(end_dim=1)
                    })

                    data_lst = []
                    for i in range(embedded_pc_0.shape[0]):
                        data = Data(x=torch.stack((embedded_pc_0, embedded_pc_1, embedded_pc_2), dim=1)[i], edge_index=torch.tensor([[1,0],[2,1]]).to("cuda:0"), edge_attribute=relationship.flatten(end_dim=1)[i])
                        data_lst.append(data)
                    # Create a batch
                    batch = Batch.from_data_list(data_lst)
                    gg = nets['graph_convolution'](batch.x, batch.edge_index, batch.edge_attribute, batch.batch)

                    gg = gg.reshape(*nagent_pos.shape[:2],-1)

                    language_embedding = nets['mlp_language'](instruction)
  
                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([gg, nagent_pos,language_embedding], dim=-1)
                    obs_cond = obs_features.flatten(start_dim=1)
                    # (B, obs_horizon * obs_dim)

                    # sample noises to add to actions
                    noise= torch.randn(naction.shape, device=device)
                    
                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                            0, noise_schedulers["single"].config.num_train_timesteps,
                            (B,), device=device).long()
                    
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    # (this is the forward diffusion process)
                    noisy_actions = noise_schedulers["single"].add_noise(
                        naction, noise, timesteps)
                    
                    # predict the noise residual
                    noise_pred = nets["invariant"](noisy_actions, timesteps, global_cond=obs_cond)

                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    lr_scheduler.step()

                    ema.step(nets.parameters())

                    # logging
                    loss_cpu = loss.item()
                    if config['wandb']:
                        wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
            tglobal.set_postfix(loss=np.mean(epoch_loss))
            # save upon request
            if (epoch_idx in [150, 200, num_epochs]) or (epoch_idx == 1):

                # create new checkpoint
                checkpoint_dir = '{}/checkpoint_epoch_{}'.format(models_save_dir, epoch_idx)

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                save(ema, nets, checkpoint_dir)


if __name__ == "__main__":
    main()