import os
import gym
import random
import numpy as np
import torch
import wandb
#from ray.util import pdb
#import ray
import time
from torch.utils.tensorboard import SummaryWriter





import gc
import torch
## MEM utils ##
def mem_report():
    '''Report the memory usage of the tensor.storage in pytorch
    Both on CPUs and GPUs are reported'''

    def _mem_report(tensors, mem_type):
        '''Print the selected tensors of type
        There are two major storage types in our major concern:
            - GPU: tensors transferred to CUDA devices
            - CPU: tensors remaining on the system memory (usually unimportant)
        Args:
            - tensors: the tensors of specified type
            - mem_type: 'CPU' or 'GPU' in current implementation '''
        print('Storage on %s' %(mem_type))
        print('-'*LEN)
        total_numel = 0
        total_mem = 0
        visited_data = []
        for tensor in tensors:
            if tensor.is_sparse:
                continue
            # a data_ptr indicates a memory block allocated
            data_ptr = tensor.storage().data_ptr()
            if data_ptr in visited_data:
                continue
            visited_data.append(data_ptr)

            numel = tensor.storage().size()
            total_numel += numel
            element_size = tensor.storage().element_size()
            mem = numel*element_size /1024/1024 # 32bit=4Byte, MByte
            total_mem += mem
            element_type = type(tensor).__name__
            size = tuple(tensor.size())

            print('%s\t\t%s\t\t%.2f' % (
                element_type,
                size,
                mem) )
        print('-'*LEN)
        print('Total Tensors: %d \tUsed Memory Space: %.2f MBytes' % (total_numel, total_mem) )
        print('-'*LEN)

    LEN = 65
    print('='*LEN)
    objects = gc.get_objects()
    print('%s\t%s\t\t\t%s' %('Element type', 'Size', 'Used MEM(MBytes)') )
    tensors = [obj for obj in objects if torch.is_tensor(obj)]
    cuda_tensors = [t for t in tensors if t.is_cuda]
    host_tensors = [t for t in tensors if not t.is_cuda]
    _mem_report(cuda_tensors, 'GPU')
    _mem_report(host_tensors, 'CPU')
    print('='*LEN)

def locate(device, *args):
    lst = []
    for item in args:
        if item is None:
            lst.append(None)
        else:
            lst.append(item.to(device))
    return lst

def parallelEval(agents, func, args):
    """
    expects a list of dicts
    """
    results = []
    for i, arg in enumerate(args):
        agent = agents[i]
        remote = getattr(agent, func).remote
        result = remote(**arg)
        results.append(result)
    results = ray.get(results)
    return results

def sequentialEval(agents, func, args):
    """
    expects a list of dicts
    """
    results = []
    for i, arg in enumerate(args):
        agent = agents[i]
        instance_func = getattr(agent, func)
        result = instance_func(**arg)
        results.append(result)
    return results

def gather(k):
    def _gather(tensor):
        """ 
        for multiple agents aligned along an axis to collect information from their k-hop neighbor
        input: [b, n_agent, dim], returns [b, n_agent, dim*n_reception_field]
        action is an one-hot embedding
        
        the first is local
        """
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape

        result = torch.zeros((b, n, (1+2*k)*depth), dtype = tensor.dtype, device=tensor.device)
        for i in range(n):
            for j in range(i-k, i+k+1):
                if j<0 or j>=n:
                    continue
                start = (j-i +1 +2*k)%(1+2*k)
                result[:, i, start*depth: start*depth+depth] = tensor[:, j]
        return result
    if k > 0:
        return _gather
    else:
        return lambda x: x
    
def reduce(k):
    """Notice that is is sum instead of mean"""
    def _reduce(tensor):
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape

        result = torch.zeros((b, n, depth), dtype = tensor.dtype, device=tensor.device)
        for i in range(n):
            for j in range(i-k, i+k+1):
                if j<0 or j>=n:
                    continue
                result[:, i] += tensor[:, j]
        return result
    if k > 0:
        return _reduce
    else:
        return lambda x: x
    
def gather2D(shape, k):
    def _gather(tensor):
        l = 1+2*k
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        tensor = tensor.view(b, shape[0], shape[1], depth)

        result = torch.zeros((b, n, l*l*depth), dtype = tensor.dtype, device=tensor.device)
        
        for x in range(shape[0]):
            for y in range(shape[1]):
                for x1 in range(x-k, x+k+1):
                    if x1<0 or x1>=shape[0]:
                        continue
                    for y1 in range(y-k, y+k+1):
                        if y1<0 or y1>=shape[1]:
                            continue
                        start = (x1-x)*shape[1]+ (y1-y)
                        start = (start+l*l) % (l*l)
                        result[:, x*shape[1]+y, start*depth: start*depth+depth] = tensor[:, x1, y1]
        return result
    if k > 0:
        return _gather
    else:
        return lambda x: x
    
def reduce2D(shape, k):
    def _reduce(tensor):
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        tensor = tensor.view(b, shape[0], shape[1], depth)

        result = torch.zeros((b, n, depth), dtype = tensor.dtype, device=tensor.device)
        
        for x in range(shape[0]):
            for y in range(shape[1]):
                for x1 in range(x-k, x+k+1):
                    if x1<0 or x1>=shape[0]:
                        continue
                    for y1 in range(y-k, y+k+1):
                        if y1<0 or y1>=shape[1]:
                            continue
                        result[:, x*shape[1]+y] += tensor[:, x1, y1]
        return result
    if k > 0:
        return _reduce
    else:
        return lambda x: x
    
def collectGraph(method, adjacency):
    """
    method = gather or reduce
    """
    adjacency = adjacency > 0
    n = adjacency.shape[0]
    def _collectGraph(tensor):
        if len(tensor.shape) == 2: # discrete action
            tensor = tensor.unsqueeze(-1)
        b, n, depth = tensor.shape
        degree = adjacency.sum(axis=1).max()
        if method == 'reduce':
            result = torch.zeros((b, n, depth), dtype = tensor.dtype, device=tensor.device)
        else:
            result = torch.zeros((b, n, depth*degree), dtype = tensor.dtype, device=tensor.device)
        for i in range(n):
            cnt = 0
            for j in range(n):
                j = (i+j)%n
                if adjacency[i, j]:
                    if method == 'reduce':
                        result[:, i] += tensor[:, j]
                    else:
                        result[:, i, cnt*depth:(cnt+1)*depth] = tensor[:, j]
                        cnt += 1
        return result
    if adjacency.sum() == adjacency.shape[0]:
        return lambda x: x
    else:
        return _collectGraph
    
    
def collect(dic={}):
    """
    selects a different gather radius (more generally, collective operation) for each data key
    the wrapper inputs raw, no redundancy data from the env
    outputs a list containing data for each agent
    """
    def wrapper(data):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                if key in dic:
                    data[key] = dic[key](data[key])
                elif "*" in dic:
                    data[key] = dic["*"](data[key])
        return dictSplit(data)
    return wrapper

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def dictSelect(dic, idx, dim=1):
    result = {}
    assert dim == 0 or dim ==1
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            if dim == 0 or dic[key].dim() < 2:
                result[key] = dic[key][idx]
            else:
                result[key] = dic[key][:, idx]
        elif isinstance(dic[key], torch.nn.ModuleList):
            result[key] = dic[key][idx]
        else:
            result[key] = dic[key]
            
    return result

def dictSplit(dic, dim=1):
    """
        gathers every tensor and modulelist
        others are broadcasted
    """
    results = []
    assert dim == 0 or dim ==1
    for key in dic:
        if isinstance(dic[key], torch.Tensor):
            length = dic[key].shape[dim]
            break
    for i in range(length):
        tmp = dictSelect(dic, i, dim)
        results.append(tmp)
    return results

def listStack(lst, dim=1):
    """ 
    takes a list (agent parallel) of lists (return values) and stacks the outer lists
    """
    results = []
    for i in range(len(lst[0])):
        results.append(torch.stack([agent_return[i] for agent_return in lst], dim=dim))
    return results

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def exists_or_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    else:
        return True

class Config(object):
    def __init__(self):
        return None


    def _toDict(self, recursive=False):
        """
            converts to dict for **kwargs
            recursive for logging
        """
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('_') and not name.endswith('_'):
                if isinstance(value, Config) and recursive:
                    value = value._toDict(recursive)
                pr[name] = value
        return pr
    
class LogClient(object):
    """
    A logger wrapper with buffer for visualized logger backends, such as tb or wandb
    counting
        all None valued keys are counters
        this feature is helpful when logging from model interior
        since the model should be step-agnostic
    Sets seed for each process
    Centralized saving
    economic logging
        stores the values, log once per log_period
    syntactic sugar
        supports both .log(data={key: value}) and .log(key=value) 
    multiple backends
        forwards logging to both tensorboard and wandb
    logger hiearchy and multiagent multiprocess logger
        the prefix does not end with "/"
        prefix = "" is the root logger
        prefix = "*/agent0" ,... are the agent loggers
        children get n_interaction from the root logger
    """
    def __init__(self, server, prefix=""):
        self.buffer = {}
        if isinstance(server, LogClient):
            prefix = f"{server.prefix}/{prefix}"
            server = server.server
        self.server = server
        self.prefix = prefix
        self.log_period = server.getArgs().log_period
        self.last_log = 0
        setSeed(server.getArgs().seed)
        
    def child(self, prefix=""):
        return LogClient(self, prefix)
        
    def flush(self):
        self.server.flush(self)
        self.last_log = time.time()
        
    def log(self, raw_data=None, **kwargs):
        if raw_data is None:
            raw_data = {}
        raw_data.update(kwargs)
        
        data = {}
        for key in raw_data: # also logs the mean for histograms
            data[key] = raw_data[key]
            if isinstance(data[key], torch.Tensor) and len(data[key].shape)>0 or\
            isinstance(data[key], np.ndarray) and len(data[key].shape)> 0:
                data[key+'_mean'] = data[key].mean()
            
        # updates the buffer
        for key in data:
            if data[key] is None:
                if not key in self.buffer:
                    self.buffer[key] = 0
                self.buffer[key] += 1
            else:
                valid = True
                # check nans
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].detach().cpu()
                    if torch.isnan(data[key]).any():
                        valid = False
                elif np.isnan(data[key]).any():
                    valid = False
                if not valid:
                    print(f'{key} is nan!')
                   # pdb.set_trace()
                    continue
                self.buffer[key] = data[key]

        # uploading
        if time.time()>self.log_period+self.last_log:
            self.flush()

    def save(self, model, info=None):
        state_dict = model.state_dict()
        state_dict = {k: state_dict[k].cpu() for k in state_dict}
        self.server.save({self.prefix: state_dict}, info)
        
    def getArgs(self):
        return self.server.getArgs()

class LogServer(object):
    """
    We do not assume the logging backend (e.g. tb, wandb) supports multiprocess logging,
    therefore we implement a centralized log manager
    
    It should not be directly invoked, since we want to hide the implementation detail (.log.remote)
    Wrap it with prefix="" to get the root logger
    
    It also keeps track of the global step
    """
    def __init__(self, args, mute=False):
        args, algo_args = args['run_args'], args['algo_args']
        self.group = algo_args.env_fn.__name__
        self.name = args.name
        if not mute:
            run=wandb.init(
                project="RL-new",
                config={"run_args":args._toDict(recursive=True),
                        "algo_args":algo_args._toDict(recursive=True)},
                name=args.name,
                group=self.group,
            )
            self.logger = run
            self.writer = SummaryWriter(log_dir=f"runs/{self.name}")
            self.writer.add_text("config", f"{args._toDict(recursive=True)}")

        self.mute = mute
        self.args = args
        self.save_period = args.save_period
        self.last_save = time.time()
        self.state_dict = {}
        self.step = 0
        self.step_key = 'interaction'
        exists_or_mkdir(f"checkpoints/{self.name}")
        
    def getArgs(self):
        return self.args
            
    def flush(self, logger=None):
        if self.mute:
            return None
        if logger is None:
            logger = self
        buffer = logger.buffer
        data = {}
        for key in buffer:
            if key == self.step_key:
                self.step = buffer[key]
            log_key = logger.prefix+"/"+key
            while log_key[0] == '/':
                 # removes the first slash, to be wandb compatible
                log_key = log_key[1:]
            data[log_key] = buffer[key]

            if isinstance(data[log_key], torch.Tensor) and len(data[log_key].shape)>0 or\
            isinstance(data[log_key], np.ndarray) and len(data[log_key].shape)> 0:
                self.writer.add_histogram(log_key, data[log_key], self.step)
            else:
                self.writer.add_scalar(log_key, data[log_key], self.step)
            self.writer.flush()

        self.logger.log(data=data, step =self.step, commit=False)
        # "warning: step must only increase "commit = True
        # because wandb assumes step must increase per commit
        self.last_log = time.time()
        
    def save(self, state_dict=None, info=None, flush=True):
        if not state_dict is None:
            self.state_dict.update(**state_dict)
        if flush and time.time() - self.last_save >= self.save_period:
            filename = f"{self.step}_{info}.pt"
            if not self.mute:
                with open(f"checkpoints/{self.name}/{filename}", 'wb') as f:
                    torch.save(self.state_dict, f)
                print(f"checkpoint saved as {filename}")
            else:
                print("not saving checkpoints because the logger is muted")
            self.last_save = time.time()
            

def setSeed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True