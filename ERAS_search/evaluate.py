import os 
import argparse
import torch
import torch
import numpy as np
from corrupter import BernCorrupter
from read_data import DataLoader
from utils import logger_init, plot_config, gen_struct
from select_gpu import select_gpu
from base_model import BaseModel

from hyperopt_master.hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial


"""
Intialize hyper-parameters
"""
parser = argparse.ArgumentParser(description="Parser for Knowledge Graph Embedding")
#parser.add_argument('--task_dir', type=str, default='../KG_Data/FB15K237', help='the directory to dataset')
parser.add_argument('--task_dir', type=str, default='/export/data/sdiaa/KG_Data/umls', help='the directory to dataset')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--model', type=str, default='random', help='model type')
parser.add_argument('--save', type=bool, default=False, help='whether save model')
parser.add_argument('--load', type=bool, default=False, help='whether load from pretrain model')
parser.add_argument('--optim', type=str, default='adagrad', help='optimization method')
parser.add_argument('--lamb', type=float, default=0.4, help='set weight decay value')
parser.add_argument('--decay_rate', type=float, default=1.0, help='set weight decay value')
parser.add_argument('--n_dim', type=int, default=256, help='set embedding dimension')
parser.add_argument('--n_sample', type=int, default=25, help='number of negative samples')
parser.add_argument('--classification', type=bool, default=False, help='number of negative samples')
parser.add_argument('--cmpl', type=bool, default=False, help='whether use complex value or not')

parser.add_argument('--gpu', type=int, default=7, help='set gpu #')

parser.add_argument('--parrel', type=int, default=1, help='set gpu #')
parser.add_argument('--lr', type=float, default=0.7, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=1, help='number of training epochs')

parser.add_argument('--n_shared_epoch', type=int, default=500, help='')

parser.add_argument('--n_controller_epoch', type=int, default=20, help='step for controller parameters')
parser.add_argument('--n_derive_sample', type=int, default=1, help='')

#arser.add_argument('--test_batch_size', type=int, default=50, help='test batch size')


parser.add_argument('--n_batch', type=int, default=4096, help='number of training batches')
parser.add_argument('--epoch_per_test', type=int, default=10, help='frequency of testing')
parser.add_argument('--filter', type=bool, default=True, help='whether do filter in testing')
parser.add_argument('--out_file_info', type=str, default='_tune', help='extra string for the output file name')
parser.add_argument('--log_to_file', type=bool, default=False, help='log to file')
parser.add_argument('--log_dir', type=str, default='./log', help='log save dir')
parser.add_argument('--log_prefix', type=str, default='', help='log prefix')
args = parser.parse_args()


"""
main function
"""
def main(args, rela_cluster, GPU, m, n, trial, cluster_way):
    
    # set number of threads in pytorch
    torch.set_num_threads(6)
    
    # select which gpu to use
    logger_init(args)
    
    # load data
    task_dir = args.task_dir
    loader = DataLoader(task_dir)
    n_ent, n_rel = loader.graph_size()
    train_data = loader.load_data('train')
    valid_data = loader.load_data('valid')
    test_data  = loader.load_data('test')
    print("Number of train:{}, valid:{}, test:{}.".format(len(train_data[0]), len(valid_data[0]), len(test_data[0])))
    #n_train = len(train_data[0])
    #args.lamb = args.lamb * args.n_batch/n_train
    
    # set gpu
    if GPU:
        torch.cuda.set_device(args.gpu) 
    
    heads, tails = loader.heads_tails()
    
    train_data = [torch.LongTensor(vec) for vec in train_data]
    valid_data = [torch.LongTensor(vec) for vec in valid_data]
    test_data  = [torch.LongTensor(vec) for vec in test_data]
    

    if args.classification:
        valid_trip_pos, valid_trip_neg = loader.load_triplets('valid')
        test_trip_pos,  test_trip_neg  = loader.load_triplets('test')
        valid_trip_pos = [torch.LongTensor(vec).cuda() for vec in  valid_trip_pos]
        valid_trip_neg = [torch.LongTensor(vec).cuda() for vec in  valid_trip_neg]
        test_trip_pos = [torch.LongTensor(vec).cuda() for vec in  test_trip_pos]
        test_trip_neg = [torch.LongTensor(vec).cuda() for vec in  test_trip_neg]
    else:
        tester_trip_class = None
    
    corrupter = None
    
    
    if cluster_way == "pde":
        file_path = "oas_pde" + "_" + str(m) + "_" + str(n)
    elif cluster_way == "scu":
        file_path = "oas_scu"  + "_" + str(m) + "_" + str(n)
    elif cluster_way == "one_clu":
        file_path = "oas_one"  + "_" + str(m) + "_" + str(n)
        
    directory = os.path.join("results", dataset, file_path)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    os.environ["OMP_NUM_THREADS"] = "4"   
    os.environ["MKL_NUM_THREADS"] = "4"   
    
    args.out_dir = directory
    
    if cluster_way == "pde":
        args.perf_file = os.path.join(directory, dataset + '_oas_pde_' + str(m) + "_" + str(n) + "_" + str(trial) + '.txt')
    elif cluster_way == "scu":
        args.perf_file = os.path.join(directory, dataset + '_oas_scu_' + str(m) + "_" + str(n) + "_" + str(trial) + '.txt')
    
    elif cluster_way == "one_clu":
        args.perf_file = os.path.join(directory, dataset + '_oas_one_' + str(m) + "_" + str(n) + "_" + str(trial) + '.txt')
    
    
    
    
    print('output file name:', args.perf_file)
    
    def tester_val(struct, test, randint):
        return model.test_link(struct=struct, test = test, test_data=valid_data, randint = randint, n_ent=n_ent, heads=heads, tails=tails, filt=args.filter)
    
    def tester_tst(struct, test):
        return model.test_link(struct=struct, test = test, test_data=test_data, n_ent=n_ent, heads=heads, tails=tails, filt=args.filter)
    
    if args.classification:
        tester_trip_class = lambda: model.test_trip_class(valid_trip_pos, valid_trip_neg, test_trip_pos, test_trip_neg)
    else:
        tester_trip_class = None
    
    if dataset == 'WN18RR':
        args.lr = 0.47102439590590006
        #args.lamb = 5.919173541218532e-05      # searched
        #args.lamb = 1.8402163609403787e-05      # SimplE
        args.lamb = 0.0002204803280058515       # ComplEx
        args.n_batch = 512
        args.decay_rate = 0.9903840888956048
        #args.n_dim = 512 
        #args.n_epoch = 300 
        args.n_epoch = 1
        args.n_dim = 512
        args.epoch_per_test = 20 #20
        
    elif dataset == 'FB15K237':
        #args.lr = 0.0885862663108572
        #args.lamb = 0.0016177695659237597
        #args.decay_rate = 0.9931763998742731
        #args.n_batch = 256
        args.lr = 0.1783468990895745
        args.lamb = 0.0025173667237246883
        args.decay_rate = 0.9915158217372417
        args.n_batch = 512
        #args.n_dim = 2048 
        #args.n_epoch = 500 
        args.n_epoch = 1 #300
        args.n_dim = 512 #2048
        args.epoch_per_test = 15
        
    elif dataset == 'WN18':
        args.lr = 0.10926076305780041
        args.lamb = 0.0003244851835920663
        args.decay_rate = 0.9908870395744
        args.n_batch = 512 #256
        #args.n_dim = 1024 
        #args.n_epoch = 400 
        args.n_dim = 512
        args.n_epoch = 1
        args.epoch_per_test = 15
        
    elif dataset == 'FB15K':
        args.lr = 0.7040329784234945
        args.lamb = 3.49037818818688153e-5
        args.decay_rate = 0.9909065915902778
        args.n_batch = 512
        #args.n_epoch = 700 
        #args.n_dim = 2048 
        args.n_dim = 512
        args.n_epoch = 1
        args.epoch_per_test = 15
        
    elif dataset == 'YAGO':
        args.lr = 0.9513908770180219
        args.lamb = 0.00021779088577909324
        args.decay_rate = 0.9914972709145934
        args.n_batch = 512 # 2048
        #args.n_dim = 1024
        #args.n_epoch = 400
        args.n_dim = 512
        args.n_epoch = 1 #300
        args.epoch_per_test = 20

    #elif dataset == 'LDC' or '12' or '14' or '15' or 'nations' or 'umls' or 'kinship':
    else:
        args.lr = 0.47102439590590006
        args.lamb = 0.0002204803280058515      
        args.n_batch = 128
        args.decay_rate = 0.9903840888956048
        args.n_epoch = 1
        args.n_dim = 512
        args.epoch_per_test = 20


    plot_config(args)

    model = BaseModel(n_ent, n_rel, args, GPU, rela_cluster, m, n, cluster_way)
    
    
    if cluster_way == "scu":
        rewards, structs, relas = model.mm_train(train_data, valid_data, valid_control, tester_val, tester_tst, tester_trip_class)
    elif cluster_way == "pde" or "one_clu":
        rewards, structs = model.mm_train(train_data, valid_data, valid_control, tester_val, tester_tst, tester_trip_class)
    
    
    
    rewards = torch.Tensor(rewards).tolist()
    structs = [item.tolist() for item in structs]
    with open(args.perf_file, 'a') as f:
        f.write("rewards:"+ str(rewards) +"\n")
        f.write("structs:"+ str(structs) +"\n")
        if cluster_way == "scu":
            f.write("rela clusters:"+ str(relas) +"\n")
        
    
    return rewards, structs


if __name__ == '__main__':
    
    # utilize GPU or not
    GPU = True

    # set data set and its corresponding path
    dataset = "FB15K"
    
    args.task_dir = "../KG_Data/" + dataset
#    if GPU:
#        args.task_dir = "/export/data/sdiaa/KG_Data/" + dataset
    
    # utilize valid data for updating architectures or not
    valid_control = -1
    
    #
    trial = 100
    
    
    """paper related settings"""
    n = 4

    # set the number of relation blocks
    m = 4 # please note that args.n_dim must can be divided by m

    cluster_way = "scu" # or "scu"
    
    if cluster_way == "pde":
        rela_cluster_list = {"WN18RR":[1, 1, 0, 1, 1, 0, 2, 0, 0, 1, 0],
                             "WN18":[1, 0, 2, 1, 2, 1, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2],
                             "FB15K237":[3, 1, 3, 2, 3, 3, 1, 3, 2, 2, 3, 3, 1, 3, 3, 0, 0, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 2, 3, 3, 3, 3, 2, 2, 3, 3, 3, 3, 3, 2, 2, 0, 3, 2, 2, 2, 3, 3, 3, 0, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 2, 3, 2, 0, 3, 3, 2, 3, 0, 1, 3, 3, 0, 2, 3, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 1, 3, 3, 1, 3, 3, 0, 1, 3, 2, 3, 2, 2, 3, 1, 3, 3, 2, 3, 2, 3, 2, 3, 3, 3, 0, 0, 0, 3, 2, 2, 0, 3, 0, 1, 2, 2, 3, 3, 2, 3, 0, 0, 1, 2, 3, 3, 3, 3, 3, 2, 1, 3, 1, 3, 3, 1, 3, 3, 3, 1, 3, 2, 2, 0, 3, 3, 3, 2, 3, 3, 1, 3, 2, 2, 3, 2, 3, 2, 2, 2, 1, 1, 3, 1, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 3, 3, 3, 1, 3, 3, 3, 0, 3, 2, 0, 3, 3, 3, 3, 0, 3, 2, 2, 3, 2, 3, 3, 3, 0, 1, 3, 0, 3, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3],
                             "FB15K":[0, 2, 0, 3, 0, 0, 1, 2, 0, 2, 0, 0, 0, 3, 1, 3, 1, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 2, 1, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 1, 1, 0, 3, 0, 3, 1, 0, 0, 0, 3, 1, 1, 3, 0, 1, 0, 3, 0, 0, 0, 0, 1, 1, 0, 1, 0, 3, 3, 3, 1, 1, 0, 2, 1, 1, 3, 1, 0, 3, 3, 0, 0, 0, 1, 1, 1, 3, 0, 0, 3, 1, 3, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 0, 0, 2, 1, 3, 1, 0, 3, 1, 0, 3, 0, 0, 1, 0, 0, 2, 3, 1, 0, 1, 1, 1, 2, 0, 2, 1, 1, 0, 1, 0, 3, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 3, 1, 1, 0, 0, 1, 1, 2, 0, 1, 1, 2, 0, 0, 0, 0, 2, 1, 0, 1, 3, 3, 0, 1, 1, 2, 1, 3, 0, 2, 3, 2, 0, 0, 0, 1, 3, 0, 1, 3, 0, 3, 0, 1, 0, 3, 3, 1, 1, 1, 0, 0, 3, 3, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 3, 1, 1, 2, 0, 0, 3, 0, 0, 1, 2, 2, 1, 1, 2, 0, 3, 3, 3, 0, 1, 0, 0, 1, 1, 3, 0, 3, 0, 0, 3, 0, 1, 2, 1, 0, 0, 1, 1, 1, 3, 0, 1, 0, 1, 0, 1, 1, 0, 1, 3, 1, 0, 0, 1, 0, 1, 0, 3, 2, 1, 0, 0, 2, 2, 1, 1, 0, 1, 1, 0, 3, 0, 1, 1, 1, 3, 0, 3, 1, 0, 1, 1, 1, 0, 0, 1, 3, 1, 3, 2, 1, 1, 3, 3, 3, 1, 0, 1, 1, 3, 3, 1, 1, 1, 3, 1, 3, 2, 3, 1, 1, 1, 0, 1, 0, 2, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 3, 3, 1, 0, 3, 1, 0, 3, 1, 1, 1, 0, 3, 0, 3, 0, 3, 1, 3, 3, 1, 1, 2, 0, 2, 3, 0, 3, 3, 1, 3, 2, 2, 1, 1, 3, 1, 0, 3, 1, 0, 0, 0, 3, 3, 1, 0, 1, 3, 0, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 0, 1, 1, 0, 1, 1, 1, 3, 1, 1, 1, 0, 2, 1, 1, 1, 1, 3, 0, 0, 0, 1, 0, 3, 0, 0, 1, 1, 1, 0, 1, 3, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 0, 3, 1, 3, 2, 0, 2, 3, 1, 3, 0, 3, 1, 0, 1, 3, 1, 3, 1, 1, 1, 0, 1, 3, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 3, 0, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 1, 1, 3, 2, 3, 0, 0, 2, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 0, 0, 1, 3, 1, 1, 3, 3, 3, 0, 1, 3, 0, 3, 3, 3, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 3, 1, 1, 3, 1, 1, 3, 3, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 0, 1, 1, 3, 1, 1, 0, 3, 1, 3, 3, 1, 3, 0, 3, 0, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 3, 1, 1, 1, 0, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 1, 3, 0, 1, 0, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 0, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 3, 3, 3, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 3, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 3, 1, 3, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 3, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 1, 1, 1, 1, 1, 3, 1],
                             }
        n = max(rela_cluster_list[dataset]) + 1 # the number of clusters
    
    elif cluster_way == "scu":
        n_rels = {"WN18RR":11, "WN18":18, "FB15K237":237, "FB15K":1345, "YAGO":37, "LDC":41, "12":6, "14":6, "15":6, "nations": 55, "umls":46, "kinship":25}
        rela_cluster_list = {}
        rela_cluster_list[dataset] = np.random.randint(0, n, n_rels[dataset])
    
    model = main(args, rela_cluster_list[dataset], GPU, m, n, trial, cluster_way)

                



