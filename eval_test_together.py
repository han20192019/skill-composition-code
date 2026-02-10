import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import mani_skill.envs.tasks.tabletop.two_step_my_together
import numpy as np
import torch
import torch.nn as nn
import collections
from diffusers.training_utils import EMAModel
from tqdm.auto import tqdm
from skvideo.io import vwrite
import os
import argparse
import json
from torch_geometric.data import Data, Batch
# dp defined utils
from utils import *
from models import *
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv, GATConv
from pointnet_extractor import DP3Encoder
from torch_geometric.nn import global_mean_pool
import fpsample
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import my_eval_model
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
import re
from PIL import Image
import io


from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
import cv2
import supervision as sv

import openai
import base64
from PIL import Image
from io import BytesIO


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

def encode_image(image):
    # Save the image to a bytes buffer
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # You can choose "JPEG" or other formats too
    buffered.seek(0)
    # Encode the image bytes as base64
    return base64.b64encode(buffered.read()).decode("utf-8")

openai.api_key = ...

import clip
# Load the CLIP model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
# Encode a sentence into a feature vector
def encode_text(sentence):
    #sentence = truncate_sentence(sentence)
    text = clip.tokenize([sentence]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text)
    return text_features


    
    
models_save_dir = "..."

stats = {'agent_pos': {'min': torch.tensor([-0.15829659,  0.2738431 , -0.0690495 , -2.3559928 , -0.05070972,
        2.0321393 ,  0.5672193 ,  0.00930391,  0.00960855], dtype=torch.float32, device="cuda:0"), 'max': torch.tensor([ 0.24690998,  0.932966  ,  0.07711923, -1.3707049 ,  0.06377019,
        2.8304572 ,  1.1111133 ,  0.04      ,  0.04      ], dtype=torch.float32, device="cuda:0")}, 'action': {'min': torch.tensor([-0.15884228,  0.2707608 , -0.06952421, -2.357214  , -0.05161012,
        2.026965  ,  0.5663745 , -1.        ], dtype=torch.float32, device="cuda:0"), 'max': torch.tensor([ 0.24830234,  0.9371018 ,  0.07746021, -1.369698  ,  0.06438983,
        2.8355749 ,  1.117754  ,  1.        ], dtype=torch.float32, device="cuda:0")}}



#load saved trained model
nets = nn.ModuleDict({})

vision_encoder = get_resnet()
vision_encoder = replace_bn_with_gn(vision_encoder)

nets['vision_encoder'] = vision_encoder



vision_feature_dim = 512
language_feature_dim = 512
lowdim_obs_dim = 9
obs_dim = vision_feature_dim + lowdim_obs_dim + language_feature_dim
action_dim = 8

pred_horizon = 16
obs_horizon = 2
action_horizon = 8
num_diffusion_iters = 100

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
    
#encode the edge attribute     
NUM_EDGE_FEATURES = 512
nets['graph_convolution'] = GDPModel(in_channels=128, hidden_channels=128, out_channels=512)

invariant = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)
nets['invariant'] = invariant 


mlp_language = MLP_Language(language_feat_dim=512, pc_feat_dim=512)
nets['mlp_language'] = mlp_language

nets = nets.to(device)

ema = EMAModel(
    parameters=nets.parameters(),
    power=0.75)

##################### LOADING Model and EMA #####################
for model_name, model in nets.items():
    model_path = os.path.join(models_save_dir, f"{model_name}.pth")
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

ema_nets = nets
ema_path = os.path.join(models_save_dir, f"ema_nets.pth")
model_state_dict = torch.load(ema_path)
ema.load_state_dict(model_state_dict)
ema.copy_to(ema_nets.parameters())

print("All models have been loaded successfully.")
model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
)

model_path = "liuhaotian/llava-v1.5-7b"


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "/home/kalman/Han/05_30_2025_skill_composition/preprocess_data/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "/home/kalman/Han/05_30_2025_skill_composition/preprocess_data/Grounded-Segment-Anything/groundingdino_swint_ogc.pth"

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = "/home/kalman/Han/05_30_2025_skill_composition/preprocess_data/Grounded-Segment-Anything/sam_vit_h_4b8939.pth"

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Building SAM Model and SAM Predictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH)
sam.to(device=DEVICE)
sam_predictor = SamPredictor(sam)


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


#------Begin Get planning according to VLM:------
test_index = 10

noise_scheduler = create_injected_noise(num_diffusion_iters)

max_steps = 250

env_id = "TwoStep_my_together"
num_eval_envs = 1
env_kwargs = dict(obs_mode="pointcloud", control_mode="pd_joint_pos", render_mode="rgb_array") # modify your env_kwargs here
eval_envs = gym.make(env_id, num_envs=num_eval_envs, **env_kwargs)
# add any other wrappers here
eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=False, record_metrics=True, auto_reset=False)

# evaluation loop, which will record metrics for complete episodes only
obs, _ = eval_envs.reset(seed=test_index)


# keep a queue of last 2 steps of observations
obs_deque = collections.deque(
    [obs] * obs_horizon, maxlen=obs_horizon)
# save visualization and rewards



current_img =  Image.fromarray(obs['sensor_data']['base_camera']['rgb'].cpu().numpy().squeeze())
current_img.save('check_image.png', format='PNG')
# === BUILD MESSAGE CONTENT ===
content = [{"type": "text", "text": "The task is to put the blue block into the small white box (so that the small white box only has the blue block). Can you give me the steps to do that? Give me the response in the following way: 1.xxx, 2.xxx"}]
base64_img = encode_image(current_img)
#content.append({"type": "text", "text": f"Image {count}:"})
content.append({
    "type": "image_url",
    "image_url": {
        "url": f"data:image/png;base64,{base64_img}",
        "detail": "high"  # Use "high" for closer behavior to ChatGPT web
    }
})
# === SEND TO API ===
response = openai.ChatCompletion.create(
    model="gpt-4o",
    messages=[
        {
            "role": "system",
            "content": (
                "You are ChatGPT, a helpful assistant. Answer clearly and visually reason about the input. "
                "If multiple images are shown, compare them thoughtfully as a human would."
            )
        },
        {
            "role": "user",
            "content": content
        }
    ],
    max_tokens=500,
    temperature=0.2
)
print(response["choices"][0]["message"]["content"])
return_sentence = response["choices"][0]["message"]["content"]    


sub_instruction = re.findall(r'\d+\.\s+(.*)', return_sentence)
sub_instruction = [step.strip() for step in sub_instruction if step]

total_step = len(sub_instruction) 
#------End Get planning according to VLM:------


all_reward = []
for test_index in range(46, 47):
    
    noise_scheduler = create_injected_noise(num_diffusion_iters)

    max_steps = 250
    
    env_id = "TwoStep_my_together"
    num_eval_envs = 1
    env_kwargs = dict(obs_mode="pointcloud", control_mode="pd_joint_pos", render_mode="rgb_array") # modify your env_kwargs here
    eval_envs = gym.make(env_id, num_envs=num_eval_envs, **env_kwargs)
    # add any other wrappers here
    eval_envs = ManiSkillVectorEnv(eval_envs, ignore_terminations=False, record_metrics=True, auto_reset=False)

    # evaluation loop, which will record metrics for complete episodes only
    obs, _ = eval_envs.reset(seed=test_index)
    
    
    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [eval_envs.render().cpu().numpy()[0]]

    original_agent_state = obs['agent']['qpos']
     

    rewards = list()
    done = False
    step_idx = 0
    transform = v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale=True),
                    v2.Resize(96),
                    v2.ToDtype(torch.float32, scale=True),
                ])
    tqdm._instances.clear()
    prev_stat = 0


    B = 1
    
    for sequence in range(total_step):
    
        # === BUILD MESSAGE CONTENT ===
        content = [{"type": "text", "text": "Give me the names of the objects appeared in this instruction: " + sub_instruction[sequence].lower() + ".(Only return the object names with comma between each one)"}]
        # === SEND TO API ===
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are ChatGPT, a helpful assistant. Answer clearly and visually reason about the input. "
                    )
                },
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500,
            temperature=0.2
        )
        print(response["choices"][0]["message"]["content"])
        output_text = response["choices"][0]["message"]["content"]
        

        object_name = ['robot hand']
        object_name.extend([obj.strip() for obj in output_text.split(',')])

            
        print(object_name)
        
        
        # Get instruction and language input here

        if 'remove' in sub_instruction[sequence].lower():
            language_instruction = f'remove {object_name[1]} from {object_name[2]}'
        elif 'place' in sub_instruction[sequence].lower():
            language_instruction = f'place {object_name[1]} into {object_name[2]}'
        else:
            print("no instruction")
            language_instruction = f'place {object_name[1]} from {object_name[2]}'
        
        language_instruction_embedding = encode_text(language_instruction)
        language_instruction_embedding = language_instruction_embedding / language_instruction_embedding.norm(dim=-1, keepdim=True)
        language_instruction_embedding = language_instruction_embedding[0].cpu().numpy()

        sub_task_step = 0
        while sub_task_step < 112:
                            
            print(sub_task_step)
            # stack the last obs_horizon number of observations
            language_instruction_embedding_here = torch.tensor(np.tile(language_instruction_embedding, (2,1))).to(device, dtype=torch.float32)
            
            agent_poses = torch.stack([x['agent']['qpos'][0] for x in obs_deque])
            agent_poses = agent_poses.to(device, dtype=torch.float32)

            # normalize observation
            nagent_poses = normalize_data(agent_poses, stats=stats['agent_pos'])            
            

            object_id = [0, 1, 2]
            here_segmentation_maniskill = np.concatenate([x['pointcloud']['segmentation'].cpu().numpy() for x in obs_deque])
                                
            here_segmentation = []
            for x in obs_deque:
                current_img = Image.fromarray(x['sensor_data']['base_camera']['rgb'].cpu().numpy().squeeze())
                current_img.save('check_image.png', format='PNG')
                

                SOURCE_IMAGE_PATH = "check_image.png"
                CLASSES = object_name
                BOX_THRESHOLD = 0.25
                TEXT_THRESHOLD = 0.25
                NMS_THRESHOLD = 0.8

                # load image
                image = cv2.imread(SOURCE_IMAGE_PATH)

                # detect objects
                detections = grounding_dino_model.predict_with_classes(
                    image=image,
                    classes=CLASSES,
                    box_threshold=BOX_THRESHOLD,
                    text_threshold=TEXT_THRESHOLD
                )

                nms_idx = torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy), 
                    torch.from_numpy(detections.confidence), 
                    NMS_THRESHOLD
                ).numpy().tolist()

                detections.xyxy = detections.xyxy[nms_idx]
                detections.confidence = detections.confidence[nms_idx]
                detections.class_id = detections.class_id[nms_idx]


                # convert detections to masks
                detections.mask = segment(
                    sam_predictor=sam_predictor,
                    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    xyxy=detections.xyxy
                )


                best_mask = {}
                for xyxy, mask, confidence, class_id, tracker_id, data in detections:
                    if class_id in best_mask:
                        if best_mask[class_id]["confidence"] < confidence:
                            best_mask[class_id] = {"mask": mask, "confidence": confidence}
                    else:
                        best_mask[class_id] = {"mask": mask, "confidence": confidence}

                
                temp = []
                for i in object_id:
                    if i in best_mask:
                        temp.append(best_mask[i]['mask'])
                    else:
                        temp.append(np.zeros((128, 128), dtype=bool))
                here_segmentation.append(temp)
            
            
            edge_embedding_vlm = []
            for x in obs_deque:
                
                current_img = Image.fromarray(x['sensor_data']['base_camera']['rgb'].cpu().numpy().squeeze())
                current_img.save('check_image.png', format='PNG')
                

                # === BUILD MESSAGE CONTENT ===
                content = [{"type": "text", "text": f"What is the spatial relationship between {object_name[1]} and {object_name[2]}? (Choose from 'sitting on', 'above', and 'next to')"}]
                base64_img = encode_image(current_img)
                #content.append({"type": "text", "text": f"Image {count}:"})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "high"  # Use "high" for closer behavior to ChatGPT web
                    }
                })
                # === SEND TO API ===
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are ChatGPT, a helpful assistant. Answer clearly and visually reason about the input. "
                                "If multiple images are shown, compare them thoughtfully as a human would."
                            )
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=500,
                    temperature=0.2
                )
                sentence0 = response["choices"][0]["message"]["content"]
                print(sentence0)
                
                if 'sitting' in sentence0.lower():
                    cube_box = f"{object_name[1]} is inside the {object_name[2]}"
                elif 'above' in sentence0.lower():
                    cube_box = f"{object_name[1]} is above the {object_name[2]}, grasped by robot hand"
                elif 'next to' in sentence0.lower():
                    cube_box = f"{object_name[1]} is next to the {object_name[2]}"
                else:
                    print("no such word")
                    cube_box = f"{object_name[1]} is next to the {object_name[2]}"
                
                cube_box_embedding = encode_text(cube_box)
                cube_box_embedding = cube_box_embedding / cube_box_embedding.norm(dim=-1, keepdim=True)
                cube_box_embedding = cube_box_embedding[0].cpu().numpy()


                
                # === BUILD MESSAGE CONTENT ===
                content = [{"type": "text", "text": f"Is the {object_name[1]} grasped by robot gripper? (Return 'yes' or 'no')"}]
                base64_img = encode_image(current_img)
                #content.append({"type": "text", "text": f"Image {count}:"})
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_img}",
                        "detail": "high"  # Use "high" for closer behavior to ChatGPT web
                    }
                })
                # === SEND TO API ===
                response = openai.ChatCompletion.create(
                    model="gpt-4o",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are ChatGPT, a helpful assistant. Answer clearly and visually reason about the input. "
                                "If multiple images are shown, compare them thoughtfully as a human would."
                            )
                        },
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=500,
                    temperature=0.2
                )
                sentence0 = response["choices"][0]["message"]["content"]
                print(sentence0)

                if "yes" in sentence0.lower():
                    cube_hand = f"robot hand is grasping the {object_name[1]}"
                else:
                    cube_hand = f"robot hand is not grasping the {object_name[1]}"
                
                cube_hand_embedding = encode_text(cube_hand)
                cube_hand_embedding = cube_hand_embedding / cube_hand_embedding.norm(dim=-1, keepdim=True)
                cube_hand_embedding = cube_hand_embedding[0].cpu().numpy()

                    
                edge_embedding_vlm.append([cube_box_embedding, cube_hand_embedding])
            
            edge_embedding_vlm = np.array(edge_embedding_vlm)
            edge_embedding_vlm = torch.tensor(edge_embedding_vlm).to(device, dtype=torch.float32)               

            
            here_segmentation = np.array(here_segmentation)
            here_pointcloud = np.concatenate([x['pointcloud']['xyzw'].cpu().numpy() for x in obs_deque])
            here_color =np.concatenate([x['pointcloud']['rgb'].cpu().numpy() for x in obs_deque])
            agent_transformation_matrix = [np.squeeze(x['extra']['agent_transformation_matrix']) for x in obs_deque]
            here_pointcloud = np.array([here_pointcloud[i] @ np.linalg.inv(agent_transformation_matrix[i]) for i in range(2)])
            here_pointcloud = here_pointcloud[...,:3]    

            here_pointcloud = here_pointcloud.reshape(2, 128, 128, 3)
            here_color = here_color.reshape(2, 128, 128, 3)
            

            NP = 100
            all_pointcloud = {0:[], 1:[], 2:[]}
            all_pointcloud_color = {0:[], 1:[], 2:[]}

            for i in object_id:
                all_points_save = []
                all_color_save = []
                for j in range(here_pointcloud.shape[0]):

                    if i == 0:
                        mask = (here_segmentation_maniskill[j][..., 0] == 11) | (here_segmentation_maniskill[j][..., 0] == 12) | (here_segmentation_maniskill[j][..., 0] == 13) | (here_segmentation_maniskill[j][..., 0] == 14) | (here_segmentation_maniskill[j][..., 0] == 15)  
                        mask = mask.reshape((128,128))
                    else:
                        mask = here_segmentation[j][i]
                    here_pointcloud0 = here_pointcloud[j][mask]
                    here_color0 = here_color[j][mask]


                    if here_pointcloud0.shape[0] == 0:
                        print(i)
                    elif here_pointcloud0.shape[0]>NP:
                        downsampled_index = fpsample.fps_sampling(here_pointcloud0, NP)
                        downsampled_points = here_pointcloud0[downsampled_index]
                        downsampled_colors = here_color0[downsampled_index]
                    else:
                        downsampled_points = np.tile(here_pointcloud0, (int(np.ceil(NP/here_pointcloud0.shape[0])), 1))[:NP]
                        downsampled_colors = np.tile(here_color0, (int(np.ceil(NP/here_color0.shape[0])), 1))[:NP]
                    all_points_save.append(downsampled_points)
                    all_color_save.append(downsampled_colors)
                
                all_pointcloud[i].append(np.array(all_points_save))
                all_pointcloud_color[i].append(np.array(all_color_save))

            npoint_cloud_0 = torch.squeeze(torch.tensor(all_pointcloud[0]).to(device, dtype=torch.float32))
            npoint_cloud_1 = torch.squeeze(torch.tensor(all_pointcloud[1]).to(device, dtype=torch.float32))
            npoint_cloud_2 = torch.squeeze(torch.tensor(all_pointcloud[2]).to(device, dtype=torch.float32))
            
            npoint_cloud_color_0 = torch.squeeze(torch.tensor(all_pointcloud_color[0]).to(device, dtype=torch.float32))
            npoint_cloud_color_1 = torch.squeeze(torch.tensor(all_pointcloud_color[1]).to(device, dtype=torch.float32))
            npoint_cloud_color_2 = torch.squeeze(torch.tensor(all_pointcloud_color[2]).to(device, dtype=torch.float32))

            # infer action
            with torch.no_grad():
                # get image features
                embedded_pc_0 = nets["point_cloud_vision_encoder"]({
                        "point_cloud": npoint_cloud_0,
                        "agent_pos": nagent_poses
                    })

                    
                embedded_pc_1 = nets["point_cloud_vision_encoder"]({
                    "point_cloud": npoint_cloud_1,
                    "agent_pos": nagent_poses
                })
                
                embedded_pc_2 = nets["point_cloud_vision_encoder"]({
                    "point_cloud": npoint_cloud_2,
                    "agent_pos": nagent_poses
                })
                                    
                data_lst = []

                for i in range(embedded_pc_0.shape[0]):
                    data = Data(x=torch.stack((embedded_pc_0, embedded_pc_1, embedded_pc_2), dim=1)[i], edge_index=torch.tensor([[1,0],[2,1]]).to("cuda:0"), edge_attribute=edge_embedding_vlm[i])
                    data_lst.append(data)
                # Create a batch
                batch = Batch.from_data_list(data_lst)
                gg = nets['graph_convolution'](batch.x, batch.edge_index, batch.edge_attribute, batch.batch)

                language_instruction_embedding_here = nets['mlp_language'](language_instruction_embedding_here)

                # concatenate vision feature and low-dim obs
                obs_features = torch.cat([gg, nagent_poses, language_instruction_embedding_here], dim=-1)

                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = obs_features.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Gaussian noise
                noisy_action = torch.randn((B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_nets["invariant"](
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )                   

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction #.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            
            for i in range(len(action)):
                # stepping env
                obs, _, terminated, truncated, info = eval_envs.step(action[i])

                # save observations
                obs_deque.append(obs)
                # and reward/vis
                reward = 0
                if obs['extra']['in_bin_done']:
                    reward = reward+1
                    
                if obs['extra']['out_bin_done']:
                    reward = reward+1
                    
                rewards.append(reward)
                imgs.append(eval_envs.render().cpu().numpy()[0])

                # update progress bar
                step_idx += 1
                sub_task_step += 1



        obs, _ = eval_envs.reset_agent_my(original_agent_state)
        obs_deque.append(obs)
        imgs.append(eval_envs.render().cpu().numpy()[0])

        
        obs, _ = eval_envs.reset_agent_my(original_agent_state)
        obs_deque.append(obs)
        imgs.append(eval_envs.render().cpu().numpy()[0])
        vwrite(f"fordemo_together_test_{test_index}.mp4", imgs)
    
    all_reward.append(np.max(rewards))
    print(test_index)
    print(np.mean(all_reward))
    np.save("result.npy", np.array(all_reward))








