################################################################################################################
# Authors:                                                                                                     #
# Kenny Young (kjyoung@ualberta.ca)                                                                            #
# Tian Tian(ttian@ualberta.ca)                                                                                 #
#                                                                                                              #
# python3 dqn.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time
import numpy as np

import random, numpy, argparse, logging, os, wandb

from scipy.stats import wasserstein_distance
from collections import namedtuple
from minatar import Environment
from risk_scheduling import RiskSchedule
################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 5000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.01
STEP_SIZE = 1e-4
# GRAD_MOMENTUM = 0.95
# SQUARED_GRAD_MOMENTUM = 0.95
# MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 1.0
N_QUANT = 64
RISK_LEVEL = 0


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################################################
# class QNetwork
#
# One hidden 2D conv with variable number of input channels.  We use 16 filters, a quarter of the original DQN
# paper of 64.  One hidden fully connected linear layer with a quarter of the original DQN paper of 512
# rectified units.  Finally, the output layer is a fully connected linear layer with a single output for each
# valid action.
#
################################################################################################################
class QNetwork(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetwork, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.n_actions = num_actions
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16

        self.phi = nn.Linear(1, num_linear_units, bias=False)
        self.phi_bias = nn.Parameter(torch.zeros(num_linear_units))

        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)
        
        # Output layer:
        self.output = nn.Linear(in_features=128, out_features=num_actions)

    # As per implementation instructions according to pytorch, the forward function should be overwritten by all
    def forward(self, x):
        x = f.relu(self.conv(x))

        tau = torch.rand(N_QUANT, 1) # (N_QUANT, 1)
        if RISK_LEVEL >= 0:
            tau = RISK_LEVEL + (1 - RISK_LEVEL) * tau
        else:
            tau = abs(RISK_LEVEL) * tau

        quants = torch.arange(0, N_QUANT, 1.0) # (N_QUANT,1)
        tau = tau.to(device)
        quants = quants.to(device)
        cos_trans = torch.cos(quants * tau * 3.141592).unsqueeze(2) # (N_QUANT, N_QUANT, 1)
        rand_feat = f.relu(self.phi(cos_trans).mean(dim=1) + self.phi_bias.unsqueeze(0)).unsqueeze(0) 
        x = x.view(x.size(0), -1).unsqueeze(1)
        x = x * rand_feat
        x = f.relu(self.fc_hidden(x))
        action_value = self.output(x).transpose(1, 2) # (bs, N_ACTIONS, N_QUANT)

        #calc variance
        q_var, q_mean = torch.var_mean(action_value, dim=2)
        q_var, q_mean = torch.mean(q_var.detach().cpu(),dim=1), torch.mean(q_mean.detach().cpu(),dim=1)

        return action_value, tau, q_var, q_mean



###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


################################################################################################################
# get_state
#
# Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
# unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).
#
# Input:
#   s: current state as numpy array
#
# Output: current state as tensor, permuted to match expected dimensions
#
################################################################################################################
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


################################################################################################################
# world_dynamics
#
# It generates the next state and reward after taking an action according to the behavior policy.  The behavior
# policy is epsilon greedy: epsilon probability of selecting a random action and 1 - epsilon probability of
# selecting the action with max Q-value.
#
# Inputs:
#   t : frame
#   replay_start_size: number of frames before learning starts
#   num_actions: number of actions
#   s: current state
#   env: environment of the game
#   policy_net: policy network, an instance of QNetwork
#
# Output: next state, action, reward, is_terminated
#
################################################################################################################
def world_dynamics(t, replay_start_size, num_actions, s, env, policy_net, target_net):

    # A uniform random policy is run before the learning starts
    if t < replay_start_size:
        action = torch.tensor([[random.randrange(num_actions)]], device=device)
    else:
        # Epsilon-greedy behavior policy for action selection
        # Epsilon is annealed linearly from 1.0 to END_EPSILON over the FIRST_N_FRAMES and stays 0.1 for the
        # remaining frames
        epsilon = END_EPSILON if t - replay_start_size >= FIRST_N_FRAMES \
            else ((END_EPSILON - EPSILON) / FIRST_N_FRAMES) * (t - replay_start_size) + EPSILON

        if numpy.random.binomial(1, epsilon) == 1:
            action = torch.tensor([[random.randrange(num_actions)]], device=device)
        else:
            # State is 10x10xchannel, max(1)[1] gives the max action value (i.e., max_{a} Q(s, a)).
            # view(1,1) shapes the tensor to be the right form (e.g. tensor([[0]])) without copying the
            # underlying tensor.  torch._no_grad() avoids tracking history in autograd.
            with torch.no_grad():
                policy_action = policy_net(s)[0]
                policy_q_action = policy_action.mean(dim=2)
                policy_actions, target_actions = policy_action.cpu()[0,:,:], target_net(s)[0].cpu()[0,:,:]
                w_dist = torch.zeros(num_actions)
                for n in range(num_actions):
                    w_dist[n]=wasserstein_distance(policy_actions[n], target_actions[n])
                action = (policy_q_action + w_dist.to(device)).argmax(dim=1).view(1,1)
        
            # import pdb;pdb.set_trace()
    # Act according to the action and observe the transition and reward
    reward, terminated = env.act(action)

    # Obtain s_prime
    s_prime = get_state(env.state())

    return s_prime, action, torch.tensor([[reward]], device=device).float(), torch.tensor([[terminated]], device=device)


################################################################################################################
# train
#
# This is where learning happens. More specifically, this function learns the weights of the policy network
# using huber loss.
#
# Inputs:
#   sample: a batch of size 1 or 32 transitions
#   policy_net: an instance of QNetwork
#   target_net: an instance of QNetwork
#   optimizer: centered RMSProp
#
################################################################################################################
def train(sample, policy_net, target_net, optimizer):
    # Batch is a list of namedtuple's, the following operation returns samples grouped by keys
    batch_samples = transition(*zip(*sample))
    # states, next_states are of tensor (BATCH_SIZE, in_channel, 10, 10) - inline with pytorch NCHW format
    # actions, rewards, is_terminal are of tensor (BATCH_SIZE, 1)
    states = torch.cat(batch_samples.state)
    next_states = torch.cat(batch_samples.next_state)
    actions = torch.cat(batch_samples.action)
    rewards = torch.cat(batch_samples.reward)
    is_terminal = torch.cat(batch_samples.is_terminal)
    # import pdb; pdb.set_trace()
    Q_s_a, tau_eval, q_var, q_mean = policy_net(states)
    mb_size = Q_s_a.size(0)
    Q_s_a = torch.stack([Q_s_a[i].index_select(0, actions[i]) for i in range(mb_size)]).squeeze(1)
    Q_s_a = Q_s_a.unsqueeze(2)

    Q_s_a_next, tau_next, q_var_next, q_mean_next = target_net(next_states)
    best_actions = Q_s_a_next.mean(dim=2).argmax(dim=1)
    
    Q_s_a_prime_a_prime = torch.stack([Q_s_a_next[i].index_select(0, best_actions[i]) for i in range(mb_size)]).squeeze(1)
    target = rewards + GAMMA * (1. - is_terminal.type(torch.int)) * Q_s_a_prime_a_prime
    # import pdb; pdb.set_trace()
    # Compute the target
    target = target.unsqueeze(1).detach()

    # Quantile Huber loss
    # pdb.set_trace()
    u = target.detach() - Q_s_a
    tau = tau_eval.unsqueeze(0)

    weight = torch.abs(tau - u.le(0.).float())
    loss = f.smooth_l1_loss(Q_s_a, target.detach())
    loss = torch.mean(weight * loss, dim=1).mean(dim=1)

    b_w = torch.ones_like(rewards).to(device)
    loss = torch.mean(b_w * loss)
   

    # Zero gradients, backprop, update the weights of policy_net

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return q_var, q_mean


################################################################################################################
# dqn
#
# DQN algorithm with the option to disable replay and/or target network, and the function saves the training data.
#
# Inputs:
#   env: environment of the game
#   replay_off: disable the replay buffer and train on each state transition
#   target_off: disable target network
#   output_file_name: directory and file name prefix to output data and network weights, file saved as 
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#   step_size: step-size for RMSProp optimizer
#
#################################################################################################################
def dqn(env, replay_off, target_off, output_file_name, store_intermediate_result=False, load_path=None, step_size=STEP_SIZE, wandb=None, rs=None):

    # Get channels and number of actions specific to each game
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    # Instantiate networks, optimizer, loss and buffer
    policy_net = QNetwork(in_channels, num_actions).to(device)
    replay_start_size = 0
    if not target_off:
        target_net = QNetwork(in_channels, num_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())

    if not replay_off:
        r_buffer = replay_buffer(REPLAY_BUFFER_SIZE)
        replay_start_size = REPLAY_START_SIZE

    # optimizer = optim.RMSprop(policy_net.parameters(), lr=step_size, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])

        if not target_off:
            target_net.load_state_dict(checkpoint['target_net_state_dict'])

        if not replay_off:
            r_buffer = checkpoint['replay_buffer']

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        e_init = checkpoint['episode']
        t_init = checkpoint['frame']
        policy_net_update_counter_init = checkpoint['policy_net_update_counter']
        avg_return_init = checkpoint['avg_return']
        data_return_init = checkpoint['return_per_run']
        frame_stamp_init = checkpoint['frame_stamp_per_run']

        # Set to training mode
        policy_net.train()
        if not target_off:
            target_net.train()

    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    while t < NUM_FRAMES:
        # Initialize the return for every episode (we should see this eventually increase)
        G = 0.0

        # Initialize the environment and start state
        env.reset()
        s = get_state(env.state())
        is_terminated = False
        while(not is_terminated) and t < NUM_FRAMES:
            # Generate data
            s_prime, action, reward, is_terminated = world_dynamics(t, replay_start_size, num_actions, s, env, policy_net, target_net)

            RISK_LEVEL = rs.eval(t)

            sample = None
            if replay_off:
                sample = [transition(s, s_prime, action, reward, is_terminated)]
            else:
                # Write the current frame to replay buffer
                r_buffer.add(s, s_prime, action, reward, is_terminated)

                # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
                if t > REPLAY_START_SIZE and len(r_buffer.buffer) >= BATCH_SIZE:
                    # Sample a batch
                    sample = r_buffer.sample(BATCH_SIZE)

            # Train every n number of frames defined by TRAINING_FREQ
            if t % TRAINING_FREQ == 0 and sample is not None:
                if target_off:
                    q_var, q_mean = train(sample, policy_net, policy_net, optimizer)
                else:
                    policy_net_update_counter += 1
                    q_var, q_mean = train(sample, policy_net, target_net, optimizer)

            # Update the target network only after some number of policy network updates
            if not target_off and policy_net_update_counter > 0 and policy_net_update_counter % TARGET_NETWORK_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

            G += reward.item()

            t += 1

            # Continue the process
            s = s_prime

        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        if e % 1000 == 0:
            logging.info("Episode " + str(e) + " | Return: " + str(G) + " | Avg return: " +
                         str(numpy.around(avg_return, 2)) + " | Frame: " + str(t)+" | Time per frame: " +str((time.time()-t_start)/t) )
            logging_wandb = {
                "EPS": round(EPSILON, 3),
                "Risk_level": RISK_LEVEL,
                "Return": G,
                "Moving AVG return": numpy.around(avg_return, 3),
            }
            
            for i in range(num_actions):
                logging_wandb["q_var_"+str(i)] = q_var[i]
                logging_wandb["q_mean_"+str(i)] = q_mean[i]
            wandb.log(
                logging_wandb
            )

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 1000 == 0:
            torch.save({
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': target_net.state_dict() if not target_off else [],
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_return': avg_return,
                        'return_per_run': data_return,
                        'frame_stamp_per_run': frame_stamp,
                        'replay_buffer': r_buffer if not replay_off else []
            }, output_file_name + "_checkpoint")

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))
        
    # Write data to file
    torch.save({
        'returns': data_return,
        'frame_stamps': frame_stamp,
        'policy_net_state_dict': policy_net.state_dict()
    }, output_file_name + "_data_and_weights")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", "-g", type=str)
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--save", "-s", action="store_true")
    parser.add_argument("--replayoff", "-r", action="store_true")
    parser.add_argument("--targetoff", "-t", action="store_true")
    # risk scheduling
    parser.add_argument('--scheduling', type=str, help='averse, neutral, seek2averse, seek2neutral')
    parser.add_argument('--eps_on_off', action="store_true")
    parser.add_argument('--anneal_time', type=int, help='annealing time counted on frames')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # If there's an output specified, then use the user specified output.  Otherwise, create file in the current
    # directory with the game's name.
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + args.game

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    env = Environment(args.game)



    print('Cuda available?: ' + str(torch.cuda.is_available()))

    wandb.init(config=args,project="MiniAtari", entity="risk_scheduling", group=args.game, name=args.scheduling+'_'+str(args.anneal_time))
    if args.scheduling == "neutral":
        risk_start, risk_finish = 0.0, 0.0

    elif args.scheduling == "averse":
        risk_start, risk_finish = -0.75, -0.75

    elif args.scheduling == "seek2neutral":
        risk_start, risk_finish = 1, 0

    elif args.scheduling == "seek2averse":
        risk_start, risk_finish = 1, -0.75

    rs = RiskSchedule(risk_start, risk_finish, args.anneal_time)
    RISK_LEVEL = rs.eval(0)
    dqn(env, args.replayoff, args.targetoff, file_name, args.save, load_file_path, args.alpha, wandb, rs)


if __name__ == '__main__':
    main()
