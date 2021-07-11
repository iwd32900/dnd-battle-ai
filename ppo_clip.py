"""
Derived from https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/ppo (git 038665d on Feb 7)

- MPI functionality commented out
- Dependence on `gym` removed
- PPO optimizer becomes a class that is fed buffer of data at each step
- Must supply boolean mask of "forbidden" actions when sampling
"""
import time

import numpy as np
import scipy.signal

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam, AdamW

# import gym
# from gym.spaces import Box, Discrete
# import spinup.algos.pytorch.ppo.core as core
# from spinup.utils.logx import EpochLogger
# from spinup.utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
# from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs



def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class Actor(nn.Module):

    def _distribution(self, obs, fbn):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, fbn, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs, fbn)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, fbn_eps=1e-3):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)
        # Probability of forbidden actions will be reduced by a factor of `fbn_eps`
        self.log_fbn_eps = -np.log(fbn_eps) # ~ +7

    def _distribution(self, obs, fbn):
        # Despite the name "logits", in the multiclass scenario these are actually log *probabilities*, not log odds.
        # See torch.distributions.utils.logits_to_probs()
        logits = self.logits_net(obs)
        fbn = torch.as_tensor(fbn, dtype=torch.bool)
        # Want some mechanism of reducing the probability of forbidden (disallowed) actions,
        # to zero or close to it.  Actually zeroing may cause some math problems...
        # logits[fbn] = -1e30 # = -np.inf # -np.inf causes NaNs in Categorical.entropy()
        # Log probs can all be shifted by a constant without changing probability distribution.
        logits[fbn] -= self.log_fbn_eps # reduce probability to near zero
        pi = Categorical(logits=logits)
        # Paper seems to recommend this route instead:  https://arxiv.org/abs/2006.14171
        # pi.probs[fbn] = 1e-6 # mask out forbidden (disallowed) actions, avoiding zero prob which can cause NaNs
        # pi = Categorical(probs=pi.probs) # force re-normalization of probs to sum to 1
        return pi

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPActorCritic(nn.Module):

    def __init__(self, obs_dim, act_dim,
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        # if isinstance(action_space, Box):
        #     self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        # elif isinstance(action_space, Discrete):
        #     self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)
        self.pi = MLPCategoricalActor(obs_dim, act_dim, hidden_sizes, activation)

        # build value function
        self.v  = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs, fbn):
        with torch.no_grad():
            pi = self.pi._distribution(obs, fbn)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy(), pi

    def act(self, obs):
        return self.step(obs)[0]



class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.

    For categoricals, action is a scalar (act_dim = None),
    while fbn_dim = the number of possible actions
    """

    def __init__(self, obs_dim, fbn_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32) # Observation (state)
        # the Forbidden mask -- set to True where actions are disallowed at that moment
        self.fbn_buf = np.zeros(combined_shape(size, fbn_dim), dtype=np.bool8)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32) # Action
        self.adv_buf = np.zeros(size, dtype=np.float32) # Advantage (of action in state)
        self.rew_buf = np.zeros(size, dtype=np.float32) # Reward (one-step)
        self.ret_buf = np.zeros(size, dtype=np.float32) # Return (discounted cumulative future rewards)
        self.val_buf = np.zeros(size, dtype=np.float32) # Value (of state)
        self.logp_buf = np.zeros(size, dtype=np.float32) # log-probability of selected action
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, fbn, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.fbn_buf[self.ptr] = fbn
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def reset(self):
        self.ptr, self.path_start_idx = 0, 0

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        # assert self.ptr == self.max_size    # buffer has to be full before you can get
        assert 1 <= self.ptr < self.max_size # we will return a slice
        used = slice(0, self.ptr)
        # self.ptr, self.path_start_idx = 0, 0
        self.reset()
        # the next two lines implement the advantage normalization trick
        # adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(
            obs=torch.as_tensor(self.obs_buf[used, ...], dtype=torch.float32),
            fbn=torch.as_tensor(self.fbn_buf[used, ...], dtype=torch.bool),
            act=torch.as_tensor(self.act_buf[used, ...], dtype=torch.float32),
            ret=torch.as_tensor(self.ret_buf[used],      dtype=torch.float32),
            adv=torch.as_tensor(self.adv_buf[used],      dtype=torch.float32),
            logp=torch.as_tensor(self.logp_buf[used],    dtype=torch.float32),
            )
        return data


class PPOAlgo:
    """
    Proximal Policy Optimization (by clipping),

    with early stopping based on approximate KL

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with a
            ``step`` method, an ``act`` method, a ``pi`` module, and a ``v``
            module. The ``step`` method should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``v``        (batch,)          | Numpy array of value estimates
                                           | for the provided observations.
            ``logp_a``   (batch,)          | Numpy array of log probs for the
                                           | actions in ``a``.
            ===========  ================  ======================================

            The ``act`` method behaves the same as ``step`` but only returns ``a``.

            The ``pi`` module's forward call should accept a batch of
            observations and optionally a batch of actions, and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       N/A               | Torch Distribution object, containing
                                           | a batch of distributions describing
                                           | the policy for the provided observations.
            ``logp_a``   (batch,)          | Optional (only returned if batch of
                                           | actions is given). Tensor containing
                                           | the log probability, according to
                                           | the policy, of the provided actions.
                                           | If actions not given, will contain
                                           | ``None``.
            ===========  ================  ======================================

            The ``v`` module's forward call should accept a batch of observations
            and return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``v``        (batch,)          | Tensor containing the value estimates
                                           | for the provided observations. (Critical:
                                           | make sure to flatten this!)
            ===========  ================  ======================================


        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.) Typically
            denoted by :math:`\epsilon`.

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    def __init__(self, actor_critic, # seed=0,
        # gamma=0.99, lam=0.97,
        clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
        train_pi_iters=80, train_v_iters=80,
        target_kl=0.01, logger_kwargs=dict()):

        self.clip_ratio = clip_ratio
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl

        # Special function to avoid certain slowdowns from PyTorch + MPI combo.
        # setup_pytorch_for_mpi()

        # Set up logger and save configuration
        # logger = EpochLogger(**logger_kwargs)
        # logger.save_config(locals())

        # Random seed
        # seed += 10000 * proc_id()
        # torch.manual_seed(seed)
        # np.random.seed(seed)

        # Instantiate environment
        # env = env_fn()
        # obs_dim = env.observation_space.shape
        # act_dim = env.action_space.shape

        # Create actor-critic module
        # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
        self.ac = actor_critic

        # # Sync params across processes
        # sync_params(ac)

        # # Count variables
        # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
        # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

        # # Set up experience buffer
        # local_steps_per_epoch = int(steps_per_epoch / num_procs())
        # buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        # self.pi_optimizer = AdamW(self.ac.pi.parameters(), lr=pi_lr, weight_decay=1)
        # self.vf_optimizer = AdamW(self.ac.v.parameters(), lr=vf_lr, weight_decay=1)

        # # Set up model saving
        # logger.setup_pytorch_saver(ac)


    def update(self, data):
        "`data` is the output of PPOBuffer.get()"

        # Set up function for computing PPO policy loss
        def compute_loss_pi(data):
            obs, fbn, act, adv, logp_old = data['obs'], data['fbn'], data['act'], data['adv'], data['logp']

            # Policy loss
            pi, logp = self.ac.pi(obs, fbn, act)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
            # ent = pi.entropy().mean()
            # print(f"Loss_pi = {loss_pi.item()}    Ent_pi = {ent.item()}")
            # # For Militia/Market/Moat, 0.01 has little effect and 0.10 prevents most learning
            # loss_pi -= 0.01*ent

            # Useful extra info
            approx_kl = (logp_old - logp).mean().item()
            ent = pi.entropy().mean().item()
            clipped = ratio.gt(1+self.clip_ratio) | ratio.lt(1-self.clip_ratio)
            clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
            pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

            return loss_pi, pi_info

        # Set up function for computing value loss
        def compute_loss_v(data):
            obs, ret = data['obs'], data['ret']
            return ((self.ac.v(obs) - ret)**2).mean()

        pi_l_old, pi_info_old = compute_loss_pi(data)
        pi_l_old = pi_l_old.item()
        v_l_old = compute_loss_v(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = compute_loss_pi(data)
            # kl = mpi_avg(pi_info['kl'])
            kl = pi_info['kl'] # mpi_avg() is the average over processors
            if kl > 1.5 * self.target_kl:
                # logger.log('Early stopping at step %d due to reaching max kl.'%i)
                #print('Early stopping at step %d due to reaching max kl.'%i)
                break
            loss_pi.backward()
            # mpi_avg_grads(ac.pi)    # average grads across MPI processes
            self.pi_optimizer.step()

        # logger.store(StopIter=i)

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data)
            loss_v.backward()
            # mpi_avg_grads(ac.v)    # average grads across MPI processes
            self.vf_optimizer.step()

        # # Log changes from update
        # kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        # logger.store(LossPi=pi_l_old, LossV=v_l_old,
        #              KL=kl, Entropy=ent, ClipFrac=cf,
        #              DeltaLossPi=(loss_pi.item() - pi_l_old),
        #              DeltaLossV=(loss_v.item() - v_l_old))

    ### Example of how to fill a PPOBuffer object:
    # # Prepare for interaction with environment
    # start_time = time.time()
    # o, ep_ret, ep_len = env.reset(), 0, 0

    # # Main loop: collect experience in env and update/log each epoch
    # for epoch in range(epochs):
    #     for t in range(local_steps_per_epoch):
    #         a, v, logp = ac.step(torch.as_tensor(o, dtype=torch.float32))

    #         next_o, r, d, _ = env.step(a)
    #         ep_ret += r
    #         ep_len += 1

    #         # save and log
    #         buf.store(o, a, r, v, logp)
    #         logger.store(VVals=v)

    #         # Update obs (critical!)
    #         o = next_o

    #         timeout = ep_len == max_ep_len
    #         terminal = d or timeout
    #         epoch_ended = t==local_steps_per_epoch-1

    #         if terminal or epoch_ended:
    #             if epoch_ended and not(terminal):
    #                 print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
    #             # if trajectory didn't reach terminal state, bootstrap value target
    #             if timeout or epoch_ended:
    #                 _, v, _ = ac.step(torch.as_tensor(o, dtype=torch.float32))
    #             else:
    #                 v = 0
    #             buf.finish_path(v)
    #             if terminal:
    #                 # only save EpRet / EpLen if trajectory finished
    #                 logger.store(EpRet=ep_ret, EpLen=ep_len)
    #             o, ep_ret, ep_len = env.reset(), 0, 0


    #     # Save model
    #     if (epoch % save_freq == 0) or (epoch == epochs-1):
    #         logger.save_state({'env': env}, None)

    #     # Perform PPO update!
    #     update()

    #     # Log info about epoch
    #     logger.log_tabular('Epoch', epoch)
    #     logger.log_tabular('EpRet', with_min_and_max=True)
    #     logger.log_tabular('EpLen', average_only=True)
    #     logger.log_tabular('VVals', with_min_and_max=True)
    #     logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
    #     logger.log_tabular('LossPi', average_only=True)
    #     logger.log_tabular('LossV', average_only=True)
    #     logger.log_tabular('DeltaLossPi', average_only=True)
    #     logger.log_tabular('DeltaLossV', average_only=True)
    #     logger.log_tabular('Entropy', average_only=True)
    #     logger.log_tabular('KL', average_only=True)
    #     logger.log_tabular('ClipFrac', average_only=True)
    #     logger.log_tabular('StopIter', average_only=True)
    #     logger.log_tabular('Time', time.time()-start_time)
    #     logger.dump_tabular()
