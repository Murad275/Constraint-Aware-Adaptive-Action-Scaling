#!/usr/bin/env python

import os
import datetime, copy
import numpy as np
import torch, math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distr
from tensorboardX import SummaryWriter
from utils import TargetNet

dirPath = os.path.dirname(os.path.realpath(__file__))



MAX_STEPS = 1000
LR_ACTS = 3e-4 
LR_VALS = 3e-4
LR_ALPHA = 3e-4
INIT_TEMPERATURE = 0.1
HID_SIZE = 256
HID_SIZE_REG = 256


MATCH_PEN = 0.01 # 0.15 humanoid current
COSTS_PEN = 20 # 30 for humanoid current
COST_THRESHOLD = 10 # 10 for humanoid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#***************************** Actor network

class SACActor(nn.Module):


    def __init__(self, obs_size, act_size):
        super(SACActor, self).__init__()
        self.mu = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size)
        )
        
        self.log_std = nn.Sequential(
            nn.Linear(obs_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, act_size),
        )
        
        self.mu.apply(self.init_weights)
        self.log_std.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):  # More robust check
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            # if m.bias is not None:
            #     m.bias.data.fill_(0.001)
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0,std=0.1)

    def forward(self, x):
        mu, log_std = self.mu(x) , self.log_std(x)
        log_std  = torch.clamp(log_std, min=-5, max=2)  
        return mu, log_std

    def sample_normal(self, state, reparameterize=True):
        mu, log_std = self.forward(state)
        std = torch.exp(log_std)        
        probs= distr.Normal(mu, std)

        if reparameterize:
            x_t = probs.rsample()
        else:
            x_t = probs.sample()

        action = torch.tanh(x_t).to(device)
        log_probs = probs.log_prob(x_t)
        log_probs -= torch.log(1-action.pow(2)+(1e-6))
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs, mu, std
#*****************************************

class TwinQNets(nn.Module):
    def __init__(self, obs_size, act_size):
        super(TwinQNets, self).__init__()

        self.q1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.q2 = nn.Sequential(
            nn.Linear(obs_size +  act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.q1.apply(self.init_weights)
        self.q2.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):  # More robust check
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            # if m.bias is not None:
            #     m.bias.data.fill_(0.001)
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0,std=0.1)

    def forward(self, obs, act):
        xtot = torch.cat([obs, act], dim=1)
        return self.q1(xtot), self.q2(xtot)
#**********************************
class TwinCNets(nn.Module):
    def __init__(self, obs_size, act_size):
        super(TwinCNets, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(obs_size + act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.c2 = nn.Sequential(
            nn.Linear(obs_size +  act_size, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, HID_SIZE),
            nn.ReLU(),
            nn.Linear(HID_SIZE, 1)
        )

        self.c1.apply(self.init_weights)
        self.c2.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):  # More robust check
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            # if m.bias is not None:
            #     m.bias.data.fill_(0.001)
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0,std=0.1)

    def forward(self, obs, act):
        xtot = torch.cat([obs, act], dim=1)
        return self.c1(xtot), self.c2(xtot)

#**********************************
class SACREG(nn.Module):
    def __init__(self, obs_size, act_size):
        super(SACREG, self).__init__()
        self.reg = nn.Sequential(
            nn.Linear(obs_size +act_size + 1, HID_SIZE_REG),
            nn.ReLU(),
            nn.Linear(HID_SIZE_REG, HID_SIZE_REG),
            nn.ReLU(),
            nn.Linear(HID_SIZE_REG, act_size)
        )
        
        self.reg.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):  # More robust check
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            # if m.bias is not None:
            #     m.bias.data.fill_(0.001)
            # nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias, mean=0,std=0.1)

    def forward(self, x, act, cost):
        xtot = torch.cat([torch.cat([x, act], dim=1), cost], dim=1)
        return  torch.sigmoid(self.reg(xtot))
#**********************************
class SAC(object):
    def __init__(self, seed, state_dim, action_dim, replay_buffer,  batch_size, name, env_id, date, discount = 0.99, reward_scale=5):
        

        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.actor = SACActor(state_dim, action_dim).to(device)
        self.qnets = TwinQNets(state_dim,action_dim).to(device)
        self.cnets = TwinCNets(state_dim,action_dim).to(device)
        self.reg = SACREG(state_dim, action_dim).to(device)
        self.qs_tgt = TargetNet(self.qnets)
        self.cs_tgt = TargetNet(self.cnets)

        self.log_alpha = torch.tensor(np.log(INIT_TEMPERATURE)).to(device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_dim
        self.action_dim = action_dim
        self.act_opt = optim.AdamW(self.actor.parameters(), lr=LR_ACTS)
        self.qnets_opt = optim.AdamW(self.qnets.parameters(), lr=LR_VALS)
        self.cnets_opt = optim.AdamW(self.cnets.parameters(), lr=1e-3)
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        self.reg_opt = optim.AdamW(self.reg.parameters(), lr=LR_ACTS)

        self.discount = discount
        self.reward_scale = reward_scale
        self.rb_states = replay_buffer
        self.train_step = 0
        self.discount_costs = 0.9
        self.writer = SummaryWriter(dirPath+'/runs/'+ "RUN_/"+str(env_id)+"_/"+str(seed)+"_/"+str(date)+"/SAC_losses"+name+"_"+str(MATCH_PEN)+"_"+str(COSTS_PEN)+ \
                                    "_sig_regs_only_"+str(HID_SIZE_REG) + "_reward_scale_" + str(reward_scale)+ "_cost_threshold_"+str(COST_THRESHOLD)+"_disc_"+str(self.discount_costs))
        self.batch_size = batch_size

    def select_action(self, states,  eval=False):
        states_v = torch.FloatTensor(states).to(device).unsqueeze(0)
        if eval == False:
            actions_v,_,_,_ = self.actor.sample_normal(states_v, reparameterize=True)
        else:
            _,_,actions_v,_= self.actor.sample_normal(states_v, reparameterize=False)
            actions_v = torch.tanh(actions_v)
        # get max cost from the cnets
        c1, c2 = self.cnets(states_v, actions_v)
        cost = torch.clip(torch.max(c1, c2), 0, COST_THRESHOLD)
        reg = self.reg(states_v, actions_v, cost)
        if self.train_step % 1000 == 0 and self.train_step > 0:
            self.writer.add_scalar('reg/train_step',torch.mean(reg), self.train_step)
        actions_v  = actions_v * reg
        # actions_v = torch.clamp(actions_v,-1,1)
        actions = actions_v.detach().cpu().numpy()
        return actions.squeeze(0), cost.squeeze(0).detach().cpu().numpy()
    #*********************************************
    def train(self, ):       
        self.train_step += 1

        self.state, self.action, self.next_state, self.reward, self.not_done, self.cost = self.rb_states.sample(self.batch_size)
        
        """
        Policy and Alpha Loss
        """
        actions_new_act, log_probs_new,_, _ = self.actor.sample_normal(self.state, reparameterize=True)

        alpha_loss = (self.log_alpha.exp() * (-log_probs_new - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        c1_new, c2_new = self.cnets(self.state, actions_new_act)
        new_costs = torch.clip(torch.max(c1_new, c2_new), 0, COST_THRESHOLD)
        regs = self.reg(self.state, actions_new_act.detach(), new_costs.detach())
        q1_val, q2_val = self.qnets(self.state, actions_new_act* regs.detach())

        min_val = torch.min(q1_val, q2_val)
        actor_loss = ((self.log_alpha.exp().detach()) * log_probs_new - min_val).mean()

        self.act_opt.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor.parameters(),0.1)
        self.act_opt.step()
        # if self.train_step % 100 == 0:
        #     self.writer.add_scalar('actor_losses/train_step',actor_loss, self.train_step)


        for _ in range (1):
            regs = self.reg(self.state, actions_new_act.detach(), new_costs.detach())
            c1_new, c2_new = self.cnets(self.state, actions_new_act.detach() * regs)
            new_costs_t = torch.clip(torch.max(c1_new, c2_new), 0, COST_THRESHOLD)
            loss_reg_1 = torch.mean(new_costs_t) 
            loss_reg_2 = torch.mean(torch.log(regs + 1e-6 ))
            loss_reg = COSTS_PEN*loss_reg_1 - MATCH_PEN * loss_reg_2
            # if self.train_step % 100 == 0:
            #     self.writer.add_scalar('loss_reg_costs/train_step',loss_reg_1, self.train_step)
            #     self.writer.add_scalar('loss_reg_penalties/train_step',loss_reg_2, self.train_step)
            self.reg_opt.zero_grad()
            loss_reg.backward()
            self.reg_opt.step()

        # for _ in range (4):
        next_actions_smpld, new_log_probs, _,_ = self.actor.sample_normal(self.next_state, reparameterize=False)
        c1_va,c2_va = self.cnets(self.next_state, next_actions_smpld)
        new_costs = torch.clip(torch.max(c1_va, c2_va), 0, COST_THRESHOLD)
        new_regs = self.reg(self.next_state, next_actions_smpld, new_costs)
        next_actions_smpld = next_actions_smpld * new_regs.detach()
        """
        QF Loss
        """
        q1_v, q2_v = self.qnets(self.state, self.action)#view(-1)
        q1, q2 = self.qs_tgt.target_model(self.next_state, next_actions_smpld) 
        target_q = torch.min(q1, q2) - (self.log_alpha.exp().detach()) * new_log_probs
        ref_q = self.reward_scale * self.reward+ self.not_done  * self.discount * target_q
        ref_q_v  = ref_q.to(device)
        #*********** calculate losses
        q1_loss_v = F.mse_loss(q1_v, ref_q_v.detach())
        q2_loss_v = F.mse_loss(q2_v, ref_q_v.detach())
        q_loss_v = q1_loss_v + q2_loss_v
        """
        CF Loss
        """
        c1_v, c2_v = self.cnets(self.state, self.action)#view(-1)
        c1, c2 = self.cs_tgt.target_model(self.next_state, next_actions_smpld)
        ref_c =  self.cost +  self.discount_costs * torch.clip(torch.min(c1, c2), 0, COST_THRESHOLD)
        ref_c_v  = ref_c.to(device)
        #*********** calculate losses
        c1_loss_v = F.mse_loss(c1_v, ref_c_v.detach())
        c2_loss_v = F.mse_loss(c2_v, ref_c_v.detach())
        c_loss_v = c1_loss_v + c2_loss_v
        #***************** update networks



        self.qnets_opt.zero_grad()
        q_loss_v.backward()
        # torch.nn.utils.clip_grad_norm_(self.qnets.parameters(),1)
        self.qnets_opt.step()
        # if self.train_step % 100 == 0:
        #     self.writer.add_scalar('q_losses/train_step',q_loss_v, self.train_step)
        self.qs_tgt.alpha_sync(alpha=1 - 5e-3)  

        self.cnets_opt.zero_grad()
        c_loss_v.backward()
        # torch.nn.utils.clip_grad_norm_(self.cnets.parameters(),1)
        self.cnets_opt.step()
        # if self.train_step % 100 == 0:
        #     self.writer.add_scalar('c_losses/train_step',c_loss_v, self.train_step)
        self.cs_tgt.alpha_sync(alpha=1 - 5e-3)
  

        return self.train_step
        

    
    
    # def save(self, filename):
    #     torch.save(self.qnets.state_dict(), filename + "_qnets")
    #     torch.save(self.qnets_opt.state_dict(), filename + "_qnets_optim")
    #     torch.save(self.log_alpha, filename + "_log_alpha")
    #     torch.save(self.log_alpha_optimizer.state_dict(), filename + "_log_alpha_optimizer")

    #     torch.save(self.actor.state_dict(), filename + "_actor")
    #     torch.save(self.act_opt.state_dict(), filename + "_actor_optimizer")

 

    # def load(self, filename):
    #     self.qnets.load_state_dict(torch.load(filename + "_qnets"))
    #     self.qnets_opt.load_state_dict(torch.load(filename + "_qnets_optim"))
    #     self.tgt_crt_net = copy.deepcopy(self.qnets)

    #     self.actor.load_state_dict(torch.load(filename + "_actor"))
    #     self.act_opt.load_state_dict(torch.load(filename + "_actor_optimizer"))
        
