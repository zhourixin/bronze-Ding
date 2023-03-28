import torch.nn as nn
import torch
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5,
                                 momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class AKG(nn.Module):
    def __init__(self, dataset, model, feature_size):
        super(AKG, self).__init__()

        self.features = nn.Sequential(*list(model.children())[:-2])
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.num_ftrs = 2048 * 1 * 1

        self.conv_block1 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        self.conv_block2 = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )
        
        self.conv_block_shape = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.conv_block_att = nn.Sequential(
            BasicConv(self.num_ftrs, feature_size, kernel_size=1, stride=1, padding=0, relu=True),
            BasicConv(feature_size, self.num_ftrs, kernel_size=3, stride=1, padding=1, relu=True)
        )

        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc2 = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc_shape = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )

        self.fc_att = nn.Sequential(
            nn.BatchNorm1d(self.num_ftrs),
            nn.Linear(self.num_ftrs, feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ELU(inplace=True),
            nn.Linear(feature_size, 512)
        )


        if dataset == 'bronze':
            self.classifier_1 = nn.Sequential(
                    nn.Linear(512, 4),
                    nn.Sigmoid()
                )
            self.classifier_2 = nn.Sequential(
                    nn.Linear(512, 11),
                    nn.Sigmoid()
                )
            self.classifier_2_1 = nn.Sequential(
                    nn.Linear(512, 11),
                )
            
            self.classifier_shape = nn.Sequential(
                    nn.Linear(512, 29),
                )
            self.classifier_shape_2 = nn.Sequential(
                nn.Linear(512, 29),
                nn.Sigmoid()
            )

            self.classifier_att = nn.Sequential(
                    nn.Linear(512, 96),
                    nn.Sigmoid()
            )


    def forward(self, x):
        x = self.features(x) 
        x_order = self.conv_block1(x) 
        x_species = self.conv_block2(x) 
        shape_feature = self.conv_block_shape(x)
        x_att = self.conv_block_att(x)

        x_order_fc = self.pooling(x_order)
        x_order_fc = x_order_fc.view(x_order_fc.size(0), -1)
        x_order_fc = self.fc1(x_order_fc)

        x_species_fc = self.pooling(x_species)
        x_species_fc = x_species_fc.view(x_species_fc.size(0), -1)
        x_species_fc = self.fc2(x_species_fc)

        shape_feature = self.pooling(shape_feature)
        shape_feature = shape_feature.view(shape_feature.size(0), -1)
        shape_feature = self.fc_shape(shape_feature)

        x_att_fc = self.pooling(x_att)
        x_att_fc = x_att_fc.view(x_att_fc.size(0), -1)
        x_att_fc = self.fc_att(x_att_fc)

        y_order_sig = self.classifier_1(self.relu(x_order_fc+x_species_fc.detach().clone()))
        y_species_sig = self.classifier_2(self.relu(x_species_fc + x_order_fc))
        y_species_sof = self.classifier_2_1(self.relu(x_species_fc + x_order_fc))
        shape_sof = self.classifier_shape(self.relu(shape_feature))
        shape_sig = self.classifier_shape_2(self.relu(shape_feature))
        y_att_sig = self.classifier_att(self.relu(x_att_fc))


        return y_order_sig, y_species_sof, y_species_sig, shape_sof, shape_sig, y_att_sig
    
