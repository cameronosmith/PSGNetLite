# This file is somewhat hackish for now, will refactor later

import os,sys

from skimage import color
import numpy as np
import tensorflow as tf # just for DeepMind's dataset
import torch
import torchvision
from torch_geometric.utils import to_dense_batch
from torch.utils.tensorboard import SummaryWriter

import PSGNetLite
import ObjectsRoomLoader

batch_size = 2
imsize     = 64

# Create train dataloader 
tf_records_path = 'datasets/objects_room_train.tfrecords'
dataset = ObjectsRoomLoader.dataset(tf_records_path, 'train')
train_dataloader = dataset.batch(batch_size)

# Should move these two functions below to another file

# From SRN utils, just formats a flattened image for image writing
def lin2img(tensor):
    batch_size, num_samples, channels = tensor.shape
    sidelen = np.sqrt(num_samples).astype(int)
    return tensor.permute(0,2,1).view(batch_size, channels, sidelen, sidelen)

# Takes the pred img and clusters produced and writes them to a TF writer
def log_imgs(pred_img,clusters,gt_img,iter_):
    
    # Write grid of output vs gt 
    grid = torchvision.utils.make_grid(
                          lin2img(torch.cat((pred_img.cpu(),gt_img.cpu()))),
                          normalize=True,nrow=batch_size)
    writer.add_image("Output_vs_GT",grid.detach().numpy(),iter_)

    # Write grid of image clusters through layers
    cluster_imgs = []
    for i,(cluster,_) in enumerate(clusters):
        for cluster_j,_ in reversed(clusters[:i+1]): cluster = cluster[cluster_j]
        pix_2_cluster = to_dense_batch(cluster,clusters[0][1])[0]
        cluster_2_rgb = torch.tensor(color.label2rgb(
                    pix_2_cluster.detach().cpu().numpy().reshape(-1,imsize,imsize) 
                                    ))
        cluster_imgs.append(cluster_2_rgb)
    cluster_imgs = torch.cat(cluster_imgs)
    grid=torchvision.utils.make_grid(cluster_imgs.permute(0,3,1,2),nrow=batch_size)
    writer.add_image("Clusters",grid.detach().numpy(),iter_)

# Create model

model=PSGNetLite.PSGNetLite(imsize)
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
 
# Logs/checkpoint paths

logging_root = "./logs"
ckpt_dir     = os.path.join(logging_root, 'checkpoints')
events_dir   = os.path.join(logging_root, 'events')
if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
if not os.path.exists(events_dir): os.makedirs(events_dir)

checkpoint_path = None
if checkpoint_path is not None:
    print("Loading model from %s" % checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path))

    
writer = SummaryWriter(events_dir)
iter_  = 0
epoch  = 0
steps_til_ckpt = 1000

# Training loop

im_loss_weight = 50

while True:
    for model_input in train_dataloader:
        
        gt_img = torch.tensor(model_input["image"].numpy()).float().cuda().view(
                                                             -1,imsize**2,3)/255

        optimizer.zero_grad()

        pred_img, clusters, all_losses = model(gt_img.view(-1,imsize,imsize,3))

        img_loss = torch.nn.functional.l1_loss(pred_img.flatten(), gt_img.flatten())
        all_losses.append({"img_loss" : im_loss_weight*img_loss})

        total_loss = 0
        for i,losses in enumerate(all_losses):
            for loss_name,loss in losses.items():
                total_loss += loss
                writer.add_scalar(str(i)+loss_name, loss, iter_)
        writer.add_scalar("total_loss", total_loss, iter_)

        total_loss.backward()
        
        if iter_ % 10 == 0:
            log_imgs(pred_img.cpu().detach(), clusters, gt_img.cpu().detach(),iter_)

        optimizer.step()

        sys.stdout.write("\rIter %07d Epoch %03d   L_img %0.4f" %
                          (iter_, epoch, img_loss))

        iter_ += 1

        if iter_ % steps_til_ckpt == 0:
            torch.save(model.state_dict(),
              os.path.join(ckpt_dir, 'epoch_%04d_iter_%06d.pth' % (epoch, iter_)))

    epoch += 1