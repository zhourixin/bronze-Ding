import torch
from utils_bronze import *
import time
from loguru import logger
from loss import FocalLoss_dating, focal_binary_cross_entropy
from evaluate_Metric import EVAL
import os


def att_Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]

def train(epoches, net, trainloader, valloader,testloader, optimizer, scheduler, CELoss, GRAPH, device, devices, save_name, EXP_number, ROOT, args):
    
    lambd_shape =  args.beta 
    lambd_shape2 = args.Lambda 

    lambda_att = args.beta
    lambda_att2 = args.Lambda
    
    max_val_acc = 0
    best_epoch = 0
    if len(devices) > 1:
        ids = list(map(int, devices))
        netp = torch.nn.DataParallel(net, device_ids=ids)
    logger_ROOT = ROOT+"/logger"
    logger.add((logger_ROOT+"/val_acc_%s.log") % str(EXP_number))
    logger.debug("===============================================================================================")
    logger.debug("EXP_name:%s, alph1 = %.5f,alph2 = %.5f,beta = %.5f, lambda = %.5f" % (args.exp_name, args.alph1, args.alph2, args.beta, args.Lambda))
    logger.debug("===============================================================================================")
    for epoch in range(epoches):
        epoch_start = time.time()
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0

        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0
        order_total = 0
        species_total= 0
        shape_correct = 0
        shape_total = 0
        attribute_acc = 0
        attribute_acc_total = 0
        idx = 0

    
        for batch_idx, (inputs, _,_,attribute_label,targets,shape_label) in enumerate(trainloader):
            idx = batch_idx
            inputs, targets = inputs.to(device), targets.to(device)
            shape_label = shape_label.to(device)
            order_targets, target_list_sig, shape_targets = get_order_family_target(targets,device, shape_label)
            
            attribute_label = attribute_label.to(device)
            optimizer.zero_grad()

            if len(devices) > 1:
                xc1_sig, xc3, xc3_sig, shape_preds, shape_sig, att_sig = netp(inputs)
            else:
                xc1_sig, xc3, xc3_sig, shape_preds, shape_sig, att_sig = net(inputs)
            att_loss = focal_binary_cross_entropy(att_sig, attribute_label)
            tree_loss, tree_loss_shape, tree_loss_att = GRAPH(torch.cat([xc1_sig, xc3_sig], 1), target_list_sig, device,torch.cat([xc1_sig, xc3_sig, shape_sig], 1),shape_targets, attribute_label,torch.cat([xc1_sig, xc3_sig, att_sig], 1))
            level_criterion_shape = FocalLoss_dating(class_num=29)
            loss_shape = level_criterion_shape(shape_preds, shape_label)
            ce_loss_species = CELoss(xc3, targets)
            loss = ce_loss_species + tree_loss + lambd_shape*loss_shape + lambd_shape2*tree_loss_shape + lambda_att*att_loss + lambda_att2*tree_loss_att

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
    
            with torch.no_grad():
                _, order_predicted = torch.max(xc1_sig.data, 1)
                order_total += order_targets.size(0)
                order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()
                _, species_predicted_soft = torch.max(xc3.data, 1)
                _, species_predicted_sig = torch.max(xc3_sig.data, 1)
                species_total += targets.size(0)
                species_correct_soft += species_predicted_soft.eq(targets.data).cpu().sum().item()
                species_correct_sig += species_predicted_sig.eq(targets.data).cpu().sum().item()
                _, shape_predicted = torch.max(shape_preds.data, 1)
                shape_total += shape_label.size(0)
                shape_correct += shape_predicted.eq(shape_label.data).cpu().sum().item()

                att_sig = torch.where(att_sig>args.sig_threshold,1,0)
                attribute_acc += att_Accuracy(attribute_label.data.cpu(), att_sig.data.cpu())
                attribute_acc_total += 1

        scheduler.step()
        train_order_acc = 100.*order_correct/order_total
        train_species_acc_soft = 100.*species_correct_soft/species_total
        train_species_acc_sig = 100.*species_correct_sig/species_total
        shape_acc = 100.*shape_correct/shape_total
        train_loss = train_loss/(idx+1)
        train_att_acc_sig = 100.*attribute_acc/attribute_acc_total
        epoch_end = time.time()
        print('Iteration %d, train_order_acc = %.5f,train_species_acc_soft = %.5f,train_species_acc_sig = %.5f, train_loss = %.6f, train_shape_acc = %.6f,train_att_acc = %.6f, Time = %.1fs' % \
            (epoch, train_order_acc, train_species_acc_soft, train_species_acc_sig, train_loss, shape_acc,train_att_acc_sig,(epoch_end - epoch_start)))

        _, val_species_acc_soft, val_species_acc_sig, _, _, _ = val(net, valloader, CELoss, GRAPH, device, args)
        
        if val_species_acc_soft > max_val_acc or val_species_acc_sig> max_val_acc:
            max_val_acc = val_species_acc_soft if val_species_acc_soft > val_species_acc_sig else val_species_acc_sig
            best_epoch = epoch
            net.cpu()
            torch.save(net, ROOT+'/bronze_pt/fold%s_model_%s.pt' % (str(EXP_number),save_name))
            net.to(device)


    print('\n\nBest Epoch: %d, Best Results: %.5f' % (best_epoch, max_val_acc))
    net = torch.load(ROOT+'/bronze_pt/fold%s_model_%s.pt' % (str(EXP_number),save_name))
    net.cuda()
    test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss, test_shape_acc, test_att_acc = test(net, testloader, CELoss, GRAPH, device, args)
    logger.debug(('final_order_acc = %.5f,final_species_acc_soft = %.5f,final_species_acc_sig = %.5f, final_loss = %.6f,final_shape_acc = %.6f,final_att_acc = %.6f' % \
                (test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss, test_shape_acc, test_att_acc)))
    logger.debug("  ")
    logger.remove()


def val(net, valloader, CELoss, GRAPH, device, Args):
    epoch_start = time.time()

    lambd_shape =  Args.beta
    lambd_shape2 = Args.Lambda

    lambda_att = Args.beta
    lambda_att2 = Args.Lambda

    with torch.no_grad():
        net.eval()
        val_loss = 0

        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        species_total= 0

        shape_correct = 0
        shape_total = 0

        attribute_acc = 0
        attribute_acc_total = 0

        idx = 0
        
        for batch_idx, (inputs, _,_,attribute_label,targets,shape_label) in enumerate(valloader):
            idx = batch_idx
            inputs, targets = inputs.to(device), targets.to(device)
            shape_label = shape_label.to(device)
            order_targets, target_list_sig, shape_targets = get_order_family_target(targets,device, shape_label)
            attribute_label = attribute_label.to(device)
            xc1_sig, xc3, xc3_sig, shape_preds, shape_sig, att_sig = net(inputs)
            att_loss = focal_binary_cross_entropy(att_sig, attribute_label)
            tree_loss, tree_loss_shape, tree_loss_att = GRAPH(torch.cat([xc1_sig, xc3_sig], 1), target_list_sig, device,torch.cat([xc1_sig, xc3_sig, shape_sig], 1),shape_targets, attribute_label,torch.cat([xc1_sig, xc3_sig, att_sig], 1))
            level_criterion_shape = FocalLoss_dating(class_num=29)
            loss_shape = level_criterion_shape(shape_preds, shape_label)
            ce_loss_species = CELoss(xc3, targets)
            loss = ce_loss_species + tree_loss + lambd_shape*loss_shape + lambd_shape2*tree_loss_shape + lambda_att*att_loss + lambda_att2*tree_loss_att

            val_loss += loss.item()
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, species_predicted_soft = torch.max(xc3.data, 1)
            _, species_predicted_sig = torch.max(xc3_sig.data, 1)
            species_total += targets.size(0)
            species_correct_soft += species_predicted_soft.eq(targets.data).cpu().sum().item()
            species_correct_sig += species_predicted_sig.eq(targets.data).cpu().sum().item()
            _, shape_predicted = torch.max(shape_preds.data, 1)
            shape_total += shape_label.size(0)
            shape_correct += shape_predicted.eq(shape_label.data).cpu().sum().item()

            att_sig = torch.where(att_sig>Args.sig_threshold,1,0)
            attribute_acc += att_Accuracy(attribute_label.data.cpu(), att_sig.data.cpu())
            attribute_acc_total += 1

        val_order_acc = 100.* order_correct/order_total
        val_species_acc_soft = 100.* species_correct_soft/species_total
        val_species_acc_sig = 100.* species_correct_sig/species_total
        shape_acc = 100.*shape_correct/shape_total
        val_loss = val_loss/(idx+1)
        epoch_end = time.time()
        val_att_acc = 100.*attribute_acc/attribute_acc_total
        print('val_order_acc = %.5f,val_species_acc_soft = %.5f,val_species_acc_sig = %.5f, val_loss = %.6f, shape_acc = %.6f,att_acc = %.6f, Time = %.4s' % \
             (val_order_acc, val_species_acc_soft, val_species_acc_sig, val_loss, shape_acc,val_att_acc,epoch_end - epoch_start))

        

    return val_order_acc, val_species_acc_soft, val_species_acc_sig, val_loss, shape_acc,val_att_acc



def test(net, testloader, CELoss, GRAPH, device, Args):
    epoch_start = time.time()

    lambd_shape =  Args.beta
    lambd_shape2 = Args.Lambda
    lambda_att = Args.beta
    lambda_att2 = Args.Lambda

    with torch.no_grad():
        net.eval()
        test_loss = 0

        order_correct = 0
        species_correct_soft = 0
        species_correct_sig = 0

        order_total = 0
        species_total= 0

        shape_correct = 0
        shape_total = 0

        attribute_acc = 0
        attribute_acc_total = 0

        idx = 0
        
        for batch_idx, (inputs, _,_,attribute_label,targets,shape_label) in enumerate(testloader):
            idx = batch_idx
            inputs, targets = inputs.to(device), targets.to(device)
            shape_label = shape_label.to(device)
            order_targets, target_list_sig, shape_targets = get_order_family_target(targets,device, shape_label)
            attribute_label = attribute_label.to(device)

            xc1_sig, xc3, xc3_sig, shape_preds, shape_sig, att_sig = net(inputs)
            att_loss = focal_binary_cross_entropy(att_sig, attribute_label)
            tree_loss, tree_loss_shape, tree_loss_att = GRAPH(torch.cat([xc1_sig, xc3_sig], 1), target_list_sig, device,torch.cat([xc1_sig, xc3_sig, shape_sig], 1),shape_targets, attribute_label,torch.cat([xc1_sig, xc3_sig, att_sig], 1))
            level_criterion_shape = FocalLoss_dating(class_num=29)
            loss_shape = level_criterion_shape(shape_preds, shape_label)
            ce_loss_species = CELoss(xc3, targets)
            loss = ce_loss_species + tree_loss + lambd_shape*loss_shape + lambd_shape2*tree_loss_shape + lambda_att*att_loss + lambda_att2*tree_loss_att
            test_loss += loss.item()

            label1_list.append(order_targets)
            label2_list.append(targets)
            predicted1_list.append(xc1_sig)
            predicted2_list.append(xc3)
    
            _, order_predicted = torch.max(xc1_sig.data, 1)
            order_total += order_targets.size(0)
            order_correct += order_predicted.eq(order_targets.data).cpu().sum().item()

            _, species_predicted_soft = torch.max(xc3.data, 1)
            _, species_predicted_sig = torch.max(xc3_sig.data, 1)
            species_total += targets.size(0)
            species_correct_soft += species_predicted_soft.eq(targets.data).cpu().sum().item()
            species_correct_sig += species_predicted_sig.eq(targets.data).cpu().sum().item()
            _, shape_predicted = torch.max(shape_preds.data, 1)
            shape_total += shape_label.size(0)
            shape_correct += shape_predicted.eq(shape_label.data).cpu().sum().item()

            att_sig = torch.where(att_sig>Args.sig_threshold,1,0)
            attribute_acc += att_Accuracy(attribute_label.data.cpu(), att_sig.data.cpu())
            attribute_acc_total += 1

        test_order_acc = 100.* order_correct/order_total
        test_species_acc_soft = 100.* species_correct_soft/species_total
        test_species_acc_sig = 100.* species_correct_sig/species_total
        test_loss = test_loss/(idx+1)
        shape_acc = 100.*shape_correct/shape_total
        epoch_end = time.time()
        test_att_acc = 100.*attribute_acc/attribute_acc_total
        print('test_order_acc = %.5f,test_species_acc_soft = %.5f,test_species_acc_sig = %.5f, test_att_acc = %.6f, test_loss = %.6f, shape_acc = %.6f, Time = %.4s' % \
             (test_order_acc, test_species_acc_soft, test_species_acc_sig, test_att_acc,test_loss, shape_acc,epoch_end - epoch_start))
        
        npy_path = os.path.join(Args.exp_path, "Metric_data")
        if os.path.exists(npy_path) is False:
            os.mkdir(npy_path)
        record_file = os.path.join(Args.exp_path, "Metric_data/final_record.csv")
        label1_list = torch.cat(label1_list, dim=0).to('cpu')
        label2_list = torch.cat(label2_list, dim=0).to('cpu')
        predicted1_list = torch.cat(predicted1_list, dim=0).to('cpu')
        predicted2_list = torch.cat(predicted2_list, dim=0).to('cpu')
        torch.save(predicted1_list, os.path.join(npy_path, "predict1.pth"))
        torch.save(label1_list, os.path.join(npy_path, "label1.pth"))
        torch.save(predicted2_list, os.path.join(npy_path, "predict2.pth"))
        torch.save(label2_list, os.path.join(npy_path, "label2.pth"))
        EVAL(npy_path, record_file)

        
    return test_order_acc, test_species_acc_soft, test_species_acc_sig, test_loss, shape_acc, test_att_acc



        
        
