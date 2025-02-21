from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d

# source: ChatGPT
class View(nn.Module):
    """Custom layer to reshape tensors inside nn.Sequential"""
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(-1, *self.shape)


# ref: mingfenl, sheldonl
class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # pass
            # TODO:
            self.decoder = torch.nn.Sequential(
                nn.Linear(512, 512 * 2 * 2 * 2),
                torch.nn.ReLU(),
                View((512, 2, 2, 2)),
                nn.ConvTranspose3d(512, 2048, kernel_size=1),
                torch.nn.ReLU(),
                View((256, 4, 4, 4)),
                nn.ConvTranspose3d(256, 1024, kernel_size=1),
                torch.nn.ReLU(),
                View((128, 8, 8, 8)),
                nn.ConvTranspose3d(128, 512, kernel_size=1),
                torch.nn.ReLU(),
                View((64, 16, 16, 16)),
                nn.ConvTranspose3d(64, 256, kernel_size=1),
                torch.nn.ReLU(),
                View((32, 32, 32, 32)),
                nn.ConvTranspose3d(32, 1, kernel_size=1),
                # torch.nn.Sigmoid()
            )
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  

            self.n_point = args.n_points
            self.decoder = torch.nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, self.n_point * 3),
            )

            # TODO:
            # self.fc_projection = nn.Linear(512, 1024)
            # self.fc1 = nn.Linear(512, 256)
            # self.fc2 = nn.Linear(256, 128)
            # self.fc3 = nn.Linear(128, 256)
            # self.fc4 = nn.Linear(256, 512)
            # self.fc_final = nn.Linear(1024, self.n_point * 3)

        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations

            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.decoder = torch.nn.Sequential(
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 1024),
                nn.ReLU(),
                nn.Linear(1024, mesh_pred.verts_packed().shape[0]*3),
            )

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            voxels_pred = self.decoder(encoded_feat)       
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # x0 = torch.nn.functional.relu(self.bn_projection(self.fc_projection(encoded_feat)))
            # x1 = torch.nn.functional.relu(self.bn1(self.fc1(x0)))
            # x2 = torch.nn.functional.relu(self.bn2(self.fc2(x1)))
            # x = torch.nn.functional.relu(self.bn3(self.fc3(x2))) + x1
            # x = torch.nn.functional.relu(self.bn4(self.fc4(x))) + x0
            # x = self.fc_final(x0)
            # x = self.fc_final(x2)
            x = self.decoder(encoded_feat)
            pointclouds_pred = x.reshape(B, self.n_point, 3)
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            deform_vertices_pred = self.decoder(encoded_feat)
            # print(deform_vertices_pred.shape)
            # print(self.mesh_pred.verts_packed().shape)
            # deform_vertices_pred = 
            # print(deform_vertices_pred.shape)
            # deform_vertices_pred =             
            result = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1, 3]))
            return  result          

