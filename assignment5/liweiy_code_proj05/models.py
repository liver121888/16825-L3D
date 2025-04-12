import torch
import torch.nn as nn
import torch.nn.functional as F


# ref: https://github.com/fxia22/pointnet.pytorch/blob/master/pointnet/model.py
class transform_net(nn.Module):
    def __init__(self, input_dim=3):
        super(transform_net, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, kernel_size=1),
            nn.BatchNorm1d(1024),
        )

        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim*input_dim)
        )

        self.input_dim = input_dim

    def forward(self, x):
        n = x.shape[0]

        x = self.encoder(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        x = self.decoder(x)

        iden = torch.eye(self.input_dim) \
            .view(1, self.input_dim*self.input_dim) \
            .expand(n, -1).to(x.device)

        x = x + iden
        return x.view(-1, self.input_dim, self.input_dim)


class feat_model(nn.Module):
    def __init__(self, global_feat = False, input_transform = False, feature_transform=False):
        super(feat_model, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = nn.Conv1d(128, 1024, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.input_transform = input_transform
        if self.input_transform:
            self.trans1 = transform_net(input_dim=3)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.trans2 = transform_net(input_dim=64)
        

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, 64)
        '''
        # pass
        x = torch.transpose(points, 2, 1)
        if self.input_transform:
            trans = self.trans1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans).transpose(2, 1)

        x = F.relu(self.bn1(self.conv1(x)))

        local_feat = x

        if self.feature_transform:
            trans = self.trans2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans).transpose(2, 1)

        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        if self.global_feat:
            return x
        elif self.feature_transform:
            x = x.unsqueeze(2).repeat(1, 1, local_feat.shape[2])
            return x, skip
        else:
            x = x.unsqueeze(2).repeat(1, 1, local_feat.shape[2])
            return torch.cat([x, local_feat], 1)

# ------ TO DO ------
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        # pass
        self.feature_extractor = feat_model(global_feat=True, 
                                            input_transform=True, 
                                            feature_transform=True)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        # pass
        # print(points.shape)
        # torch.Size([32, 10000, 3])
        x = self.feature_extractor(points)
        # print(x.shape)
        # ([32, 1024, 10000])

        x = self.decoder(x)
        return F.log_softmax(x, dim=1)

# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes = 6):
        super(seg_model, self).__init__()
        # pass
        self.feature_extractor = feat_model(global_feat=False, 
                                            input_transform=True, 
                                            feature_transform=True)
        self.decoder = nn.Sequential(
            nn.Conv1d(1024, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.layer = nn.Conv1d(256, self.num_seg_classes, 1)

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        # pass
        B, N, D = points.shape
        x, skip = self.feature_extractor(points)
        x = self.decoder(x)
        x = torch.cat([x, skip], 1)
        x = self.layer(x)
        x = F.log_softmax(torch.transpose(x, 2, 1).contiguous().view(-1,self.num_seg_classes), dim=-1)
        return x.view(B, N, self.num_seg_classes)


