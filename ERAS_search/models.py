import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import time



class KGEModule(nn.Module):
    def __init__(self, n_ent, n_rel, args, GPU, rela_cluster, m, n):
        super(KGEModule, self).__init__()
        self.n_ent = n_ent
        self.n_rel = n_rel
        
        self.args = args
        self.n_dim = args.n_dim
        self.lamb = args.lamb

        self.ent_embed = nn.Embedding(n_ent, args.n_dim)
        self.rel_embed = nn.Embedding(n_rel, args.n_dim)
        self.init_weight()
        

        self.K = m
        self.GPU = GPU
        self.rela_cluster = rela_cluster
        self.n_cluster = n
        
        
    
    def weights_to_struct(self, weights):
        
        index = np.nonzero(weights.data.cpu().numpy())
        
        struct = []
        for i, j in zip(index[0], index[1]):
            idx_head = i//4
            idx_tail = i%4
            if j == 8:
                # 
                continue
            else:
                idx_rela = j//2
                if j%2 ==0:
                    sym = 1
                else:
                    sym = -1
            struct += [idx_head,idx_tail,idx_rela,sym]
        return struct

        
        

    def init_weight(self):
        for param in self.parameters():
            nn.init.xavier_uniform_(param.data)
            
    def name(self, idx):
        i = idx[0]
        i_rc =  self.rela_cluster[i]
        self.r_embed[i,:,:] = self.rel_embed_2K_1[i,self.idx_list[i_rc],:] * self._arch_parameters[i_rc][[j for j in range(self.K*self.K)], self.idx_list[i_rc]].view(-1,1)



    def forward(self, struct, head, tail, rela, cluster_rela_dict, updateType="weights"):


        self.cluster_rela_dict = cluster_rela_dict
        
        """convert the architect into struct list"""        
        length = self.n_dim // self.K
        
        # create a rela_embed with size (n_rel, 2K+1, length)
        rel_embed_pos = self.rel_embed.weight.view(-1, self.K, length)
        rel_embed_neg = -rel_embed_pos
        
        if self.GPU:
            rel_embed_zeros = torch.zeros(self.n_rel, 1, length).cuda()
        else:
            rel_embed_zeros = torch.zeros(self.n_rel, 1, length)
            
        self.rel_embed_2K_1 = torch.cat((rel_embed_zeros, rel_embed_pos, rel_embed_neg),1)
                
        #struct = self._arch_parameters

        # combine struct
        if self.GPU:
            self.r_embed = torch.zeros(self.n_rel, self.K*self.K, length).cuda()
        else:
            self.r_embed = torch.zeros(self.n_rel, self.K*self.K, length)        

        for i_rc in range(self.n_cluster):
            #x = torch.LongTensor([i_rc for i in range(len(self.cluster_rela_dict[i_rc]))])
            max_idx_list = struct[i_rc]
            self.r_embed[self.cluster_rela_dict[i_rc],:,:] = self.rel_embed_2K_1[self.cluster_rela_dict[i_rc]][:,max_idx_list,:]
        
        self.r_embed = self.r_embed.view(-1, self.K, self.K, length)
                
        head = head.view(-1)
        tail = tail.view(-1)
        rela = rela.view(-1)
        
        head_embed = self.ent_embed(head).view(-1, self.K, self.n_dim//self.K)
        tail_embed = self.ent_embed(tail).view(-1, self.K, self.n_dim//self.K)
        rela_embed = self.r_embed[rela,:,:,:]
        
        pos_trip = self.test_trip(head_embed, rela_embed, tail_embed)
        
        neg_tail = self.test_tail(head_embed, rela_embed)
        neg_head = self.test_head(rela_embed, tail_embed)

        max_t = torch.max(neg_tail, 1, keepdim=True)[0]
        max_h = torch.max(neg_head, 1, keepdim=True)[0]

        loss = - 2*pos_trip + max_t + torch.log(torch.sum(torch.exp(neg_tail - max_t), 1)) +\
               max_h + torch.log(torch.sum(torch.exp(neg_head - max_h), 1))
        
        self.regul = torch.sum(rela_embed**2)
        

        return torch.sum(loss) #, init_time, pos_time, neg_time, loss_time, time_first, time_second, time_third
    
    def test_trip(self, head, rela, tail):
        vec_hr = self.get_hr(head, rela)
        scores = (vec_hr*tail).view(-1, self.n_dim)
        return torch.sum(scores, 1)


    def get_hr(self, head, rela):
        n_head, length = len(head), self.n_dim//self.K
        if self.GPU:
            vs = torch.zeros(n_head, self.K, length).cuda()
        else:
            vs = torch.zeros(n_head, self.K, length)
        for i in range(self.K):
            #vs += head*rela[:,:,i,:] #please pay attention to it
            vs[:,i,:] = torch.sum((head*rela[:,:,i,:]),1)
        return vs
    
    def get_rt(self, rela, tail):
        n_head, length = len(tail), self.n_dim//self.K
        if self.GPU:
            vs = torch.zeros(n_head, self.K, length).cuda()
        else:
            vs = torch.zeros(n_head, self.K, length)
        for i in range(self.K):
            #vs += tail*rela[:,:,i,:]
            #vs += tail*rela[:,i,:,:] #please pay attention to it
            vs[:,i,:] = torch.sum((tail*rela[:,i,:,:]) ,1)            
        return vs
        
    def test_tail(self, head, rela):
        #print(self.struct)
        vec_hr = self.get_hr(head, rela).view(-1,self.n_dim)
        tail_embed = self.ent_embed.weight
        scores = torch.mm(vec_hr, tail_embed.transpose(1,0))
        return scores
    
    def test_head(self, rela, tail):
        #print(self.struct)
        vec_rt = self.get_rt(rela, tail).view(-1,self.n_dim)
        head_embed = self.ent_embed.weight
        scores = torch.mm(vec_rt, head_embed.transpose(1,0))
        return scores
    
    
    def arch_parameters(self):
        return self._arch_parameters
    
    def arch_parametersTau(self):
        return self._arch_parametersTau


    def _initialize_alphas(self):

        k, num_pos = self.K**2, 2*self.K+1
        if self.GPU:
            self._arch_parameters = Variable(torch.ones(self.n_cluster, k, num_pos).cuda()/2, requires_grad=True)
        else:
            self._arch_parameters = Variable(torch.ones(self.n_cluster, k, num_pos)/2, requires_grad=True)
        
        
    def _intiialize_alphasTau(self):
        k, num_pos = self.K**2, 2*self.K+1
        
        if self.GPU:
            self._arch_parametersTau = torch.zeros(self.n_cluster, k, num_pos).cuda()
        else:
            self._arch_parametersTau = torch.zeros(self.n_cluster, k, num_pos)

        
    
    def save_params(self):
        for index, value in enumerate(self._arch_parameters):
          self.saved_params[index].copy_(value.data)
    
    def binarization(self, tau_state=False):
        if self.rand_seed > 0:
            np.random.seed(self.rand_seed)

        self.save_params()
        
        for index in range(len(self._arch_parameters)):
          m,n = self._arch_parameters[index].size()
          Tau = False
          if np.random.rand() <= self.tau and tau_state:
            maxIndexs = np.random.choice(range(n), m)
            Tau = True
          else:
            maxIndexs = self._arch_parameters[index].data.cpu().numpy().argmax(axis=1)

            
          if self.GPU:
              self._arch_parameters.data[index] = self.proximal_step(self._arch_parameters[index], index, maxIndexs)              
          else:
              self._arch_parameters[index].data = self.proximal_step(self._arch_parameters[index], index, maxIndexs)
          
          if not Tau:
              if self.GPU:
                  self._arch_parametersTau.data[index].copy_(self._arch_parameters.data[index])
              else:
                  self._arch_parametersTau[index].data.copy_(self._arch_parameters.data[index])
                  

    def restore(self):
        for index, value in enumerate(self.saved_params):
            if self.GPU:
                self._arch_parameters.data[index].copy_(self.saved_params[index].data)
            else:
                self._arch_parameters[index].data.copy_(self.saved_params[index].data)
          
    def clip(self):
        clip_scale = []
        m = nn.Hardtanh(0, 1)
        #m = nn.Softmax(dim=1) # change from hardtanh to softmax
        
        for index in range(len(self._arch_parameters)):
          if self.GPU:
              clip_scale.append(m(Variable(self._arch_parameters.data[index])))
          else:
              clip_scale.append(m(Variable(self._arch_parameters[index].data)))
              
#          if index == 0:
#              print("clip")
#              print(clip_scale[0])
              
              
        for index in range(len(self._arch_parameters)):
          if self.GPU:
              self._arch_parameters.data[index] = clip_scale[index].data
          else:
              self._arch_parameters[index].data = clip_scale[index].data
              
          
    
    def proximal_step(self, var, out_index, maxIndexs=None):
        values = var.data.cpu().numpy()
        m,n = values.shape
        
        # records initial weights
        record = np.zeros((2,n))
        record[0] = var.data.cpu().numpy().max(axis=0)
        record[1] = var.data.cpu().numpy().argmax(axis=0)    
        
        # binarization
        alphas = []
        for i in range(m):
          for j in range(n):
            if j==maxIndexs[i]:
              alphas.append(values[i][j].copy())
              values[i][j]=1
            else:
              values[i][j]=0
        
        """unknown proximal"""
        step = 2
        cur = 0
        while(cur<m):
          cur_alphas = alphas[cur:cur+step]
          reserve_index = [v[0] for v in sorted(list(zip(range(len(cur_alphas)), cur_alphas)), key=lambda x:x[1],
                                                reverse=True)[:2]]
          for index in range(cur,cur+step):
            if index == m:
              break
            if (index - cur) in reserve_index:
              continue
            else:
              values[index] = np.zeros(n)
          cur = cur + step
          step += 1
              
        
        # check exlpoit
#        for col in range(0, (n-1)//2):
#            if np.count_nonzero(values[:,col]) + np.count_nonzero(values[:,col+(n-1)//2]) == 0:
#                if record[0][col] >= record[0][col+(n-1)//2]:
#                    idx = col
#                else:
#                    idx = col+(n-1)//2
#                values[int(record[1][idx])] = np.zeros(n)
#                values[int(record[1][idx])][idx] = 1  
  

        for col in range(1, (n-1)//2+1):
            if np.count_nonzero(values[:,col]) + np.count_nonzero(values[:,col+(n-1)//2]) == 0:
                if record[0][col] >= record[0][col+(n-1)//2]:
                    idx = col
                else:
                    idx = col+(n-1)//2
                values[int(record[1][idx])] = np.zeros(n)
                values[int(record[1][idx])][idx] = 1  
      
    
        if self.GPU:
            return torch.Tensor(values).cuda()
        else:
            return torch.Tensor(values)
        
        
    
    
    def _loss(self, h, t, r, updateType, cluster_rela_dict):
        return self.forward(h, t, r, cluster_rela_dict, updateType)


    




