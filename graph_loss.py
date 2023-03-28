import torch
import torch.nn as nn



class GraphLoss(nn.Module):
    def __init__(self, hierarchy, total_nodes, levels, device, args):
        super(GraphLoss, self).__init__()
        self.stateSpace = self.generateStateSpace(hierarchy, total_nodes, levels).to(device) # [N+1, N]
        self.ShapeStateSpace = self.generateShapeStateSpace(total_nodes, self.stateSpace, shape_nodes=29).to(device)      
        self.AttStateSpace = self.generateAttStateSpace(total_nodes, self.stateSpace, att_nodes=96).to(device)

        self.alph1 = args.alph1
        self.alph2 = args.alph2
        

    def forward(self, fs, labels, device, label_fs,labels_shape, attribute_label, att_fs): 

        index = torch.mm(self.stateSpace, fs.T) 
        joint = torch.exp(index) 
        z = torch.sum(joint, dim=0) 
        loss = torch.zeros(fs.shape[0], dtype=torch.float64).to(device) 
        p_array = torch.zeros(fs.shape[0], dtype=torch.float64).to(device) 
        for i in range(len(labels)):
            marginal = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, labels[i]] > 0)[0]))
            p = marginal / z[i]
            p_array[i] = p
            loss[i] = -torch.log(p)

###############################################################################
### Focal-type Probabilistic Classification Loss
################################################################################
        index_shape = torch.mm(self.ShapeStateSpace, label_fs.T)
        joint_shape = torch.exp(index_shape)
        z_shape = torch.sum(joint_shape, dim=0)
        loss_shape = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        p_array_shape = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        for i in range(len(labels_shape)):
            marginal_shape = torch.sum(torch.index_select(joint_shape[:, i], 0, torch.where(self.ShapeStateSpace[:, labels_shape[i]] > 0)[0]))
            pt = marginal_shape / z_shape[i]
            p_array_shape[i] = pt
            loss_shape[i] = -((1-p_array[i])**self.alph1)*torch.log((pt)) 

        index_att = torch.mm(self.AttStateSpace, att_fs.T)
        joint_att = torch.exp(index_att)
        z_att = torch.sum(joint_att, dim=0)
        loss_att = torch.zeros(fs.shape[0], dtype=torch.float64).to(device)
        for i in range(len(attribute_label)):
            att_targets = attribute_label[i]
            marginal_att = 0
            for idx, j in enumerate(att_targets):
                there_att_label = idx+15
                if j > 0:
                    marginal_att += torch.sum(torch.index_select(joint_att[:, i], 0, torch.where(self.AttStateSpace[:, there_att_label] > 0)[0]))
            
            pt = marginal_att / z_att[i]
            loss_att[i] = -((1-p_array_shape[i])**self.alph2)*torch.log((pt)) 


        return torch.mean(loss), torch.mean(loss_shape), torch.mean(loss_att)

    def inference(self, fs, device):
        with torch.no_grad():
            index = torch.mm(self.stateSpace, fs.T)
            joint = torch.exp(index)
            z = torch.sum(joint, dim=0)
            pMargin = torch.zeros((fs.shape[0], fs.shape[1]), dtype=torch.float64).to(device)
            for i in range(fs.shape[0]):
                for j in range(fs.shape[1]):
                    pMargin[i, j] = torch.sum(torch.index_select(joint[:, i], 0, torch.where(self.stateSpace[:, j] > 0)[0]))
            return pMargin

    def generateStateSpace(self, hierarchy, total_nodes, levels):
        stateSpace = torch.zeros(total_nodes + 1, total_nodes)
        recorded = torch.zeros(total_nodes)
        i = 1

        if levels == 2:
            for path in hierarchy:
                if recorded[path[1]] == 0:
                    stateSpace[i, path[1]] = 1
                    recorded[path[1]] = 1
                    i += 1
                stateSpace[i, path[1]] = 1
                stateSpace[i, path[0]] = 1
                i += 1

        elif levels == 3:
            for path in hierarchy:
                if recorded[path[1]] == 0:
                    stateSpace[i, path[1]] = 1
                    recorded[path[1]] = 1
                    i += 1
                if recorded[path[2]] == 0:
                    stateSpace[i, path[1]] = 1
                    stateSpace[i, path[2]] = 1
                    recorded[path[2]] = 1
                    i += 1
                stateSpace[i, path[1]] = 1
                stateSpace[i, path[2]] = 1
                stateSpace[i, path[0]] = 1
                i += 1
            
        if i == total_nodes + 1:
            return stateSpace
        else:
            print('Invalid StateSpace!!!')

    def generateShapeStateSpace(self, total_nodes, origin_stateSpace, shape_nodes=29):

        origin_stateSpace = origin_stateSpace.cpu()
        ShapeStateSpace = []
        leaglShape = torch.zeros((1,total_nodes+shape_nodes))
        ShapeStateSpace.append(leaglShape)
        needed_index = [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]

        shape_index = 0
        for state_index in needed_index:
            legal_state = origin_stateSpace[state_index, :].unsqueeze(0)
            for i in range(shape_nodes):
                shape_state = torch.zeros((1,shape_nodes))
                shape_state[0][i] = 1
                result_state = torch.cat((legal_state, shape_state), 1)
                ShapeStateSpace.append(result_state)

            shape_index += 1
        if shape_index != 16:
            print('Invalid ShapeStateSpace!!!')
        ShapeStateSpace = torch.cat(ShapeStateSpace, 0)

        return ShapeStateSpace

    def generateAttStateSpace(self, total_nodes, origin_stateSpace, att_nodes=96):

        origin_stateSpace = origin_stateSpace.cpu()
        AttStateSpace = []
        leaglAtt = torch.zeros((1,total_nodes+att_nodes))
        AttStateSpace.append(leaglAtt)
        needed_index = [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]

        att_index = 0
        for state_index in needed_index:
            legal_state = origin_stateSpace[state_index, :].unsqueeze(0)
            for i in range(att_nodes):
                att_state = torch.zeros((1,att_nodes))
                att_state[0][i] = 1
                result_state = torch.cat((legal_state, att_state), 1)
                AttStateSpace.append(result_state)

            att_index += 1
        if att_index != 16:
            print('Invalid attStateSpace!!!')
        AttStateSpace = torch.cat(AttStateSpace, 0)

        return AttStateSpace
    