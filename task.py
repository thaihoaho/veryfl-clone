import logging
logger = logging.getLogger(__name__)

from torch.utils.data import DataLoader

from config.algorithm import Algorithm
from server.aggregation_alg.fedavg import fedavgAggregator
from client.clients import Client, BaseClient, SignClient
from client.trainer.fedproxTrainer import fedproxTrainer
from client.trainer.SignTrainer import SignTrainer
from model.ModelFactory import ModelFactory
from dataset.DatasetFactory import DatasetFactory
from dataset.DatasetSpliter import DatasetSpliter
from chainfl.interact import chain_proxy


class Task:
    '''
    WorkFlow of a Task:
    0. Construct (Model, Dataset)--> Benchmark
    1. Construct (Server, Client)--> FL Algorithm
    3. Process of the dataset 
    '''
    def __init__(self, global_args: dict, train_args: dict, algorithm: Algorithm):
        self.global_args = global_args
        self.train_args = train_args
        self.model = None
        # Determine device: prefer explicit config, otherwise use CUDA when available
        try:
            import torch

            requested_device = None
            if isinstance(self.train_args, dict):
                requested_device = self.train_args.get('device')

            if requested_device is not None:
                # cho phép 'cpu' hoặc 'cuda'
                try:
                    self.device = torch.device(requested_device)
                    # thử kiểm tra CUDA availability nếu 'cuda'
                    if self.device.type == 'cuda' and not torch.cuda.is_available():
                        print("CUDA không khả dụng, dùng CPU")
                        self.device = torch.device('cpu')
                except Exception:
                    print("Thiết lập device không hợp lệ, fallback CPU")
                    self.device = torch.device('cpu')
            else:
                # default: dùng cuda nếu có, nếu không dùng CPU
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        except ImportError:
            # nếu torch không cài được
            print("Torch không khả dụng, dùng CPU")
            try:
                print("torch not importable in this environment")
            except Exception:
                pass
            self.device = 'cpu'

        
        #Get Dataset
        # TODO pass the schema (object) instead of args directly.  
        logger.info("Constructing dataset %s from dataset Factory", global_args.get('dataset'))
        self.train_dataset = DatasetFactory().get_dataset(global_args.get('dataset'),True)
        self.test_dataset =  DatasetFactory().get_dataset(global_args.get('dataset'),False)
        # Optional: allow running on a smaller subset for quick tests
        max_train = global_args.get('max_train_samples', None)
        max_test = global_args.get('max_test_samples', None)
        if max_train is not None:
            try:
                max_train = int(max_train)
                from torch.utils.data import Subset
                max_train = min(max_train, len(self.train_dataset))
                self.train_dataset = Subset(self.train_dataset, list(range(max_train)))
                logger.info("Subsetting train dataset to %d samples", max_train)
            except Exception:
                logger.warn("Failed to subset train dataset with max_train_samples=%s", str(max_train))
        if max_test is not None:
            try:
                max_test = int(max_test)
                from torch.utils.data import Subset
                max_test = min(max_test, len(self.test_dataset))
                self.test_dataset = Subset(self.test_dataset, list(range(max_test)))
                logger.info("Subsetting test dataset to %d samples", max_test)
            except Exception:
                logger.warn("Failed to subset test dataset with max_test_samples=%s", str(max_test))
        #Get Model
        logger.info("Constructing Model from model factory with model %s and class_num %d", global_args['model'], global_args['class_num'])
        self.model = ModelFactory().get_model(model=self.global_args.get('model'),class_num=self.global_args.get('class_num'))
        
        #FL alg
        logger.info("Algorithm: {algorithm}")
        self.server = algorithm.get_server()
        self.server = self.server()
        self.trainer = algorithm.get_trainer()
        self.client = algorithm.get_client()
        
        #Get Client and Trainer
        self.client_list = None
        self.client_pool : list[Client] = []
        
    def __repr__(self) -> str:
        pass
    
    def _construct_dataloader(self):
        logger.info("Constructing dataloader with batch size %d, client_num: %d, non-iid: %s", self.global_args.get('batch_size')
                    , chain_proxy.get_client_num(), "True" if self.global_args['non-iid'] else "False")
        batch_size = self.global_args.get('batch_size')
        batch_size = 8 if (batch_size is None) else batch_size
        self.train_dataloader_list = DatasetSpliter().random_split(dataset     = self.train_dataset,
                                                                   client_list = chain_proxy.get_client_list(),
                                                                   batch_size  = batch_size)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=batch_size, shuffle=True)
    
    def _construct_sign(self):
        self.keys_dict = dict()
        self.keys = list()
        sign_num = self.global_args.get('sign_num')
        if(None == sign_num): 
            sign_num = 0
            logger.info("No client need to add watermark")
            for ind, (client_id,_) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = None
        else:
            logger.info(f"{sign_num} client(s) will inject watermark into their models")
            
            for i in range(self.global_args.get('client_num')):
                if i < self.global_args.get('sign_num'):
                    key = chain_proxy.construct_sign(self.global_args)
                    self.keys.append(key)
                else : 
                    self.keys.append(None)
            for ind, (client_id,_) in enumerate(self.client_list.items()):
                self.keys_dict[client_id] = self.keys[ind]
            #Project the watermake to the client TODO work with the blockchain
            #Get model Here better split another function.                 
            tmp_args = chain_proxy.construct_sign(self.global_args)
            self.model = ModelFactory().get_sign_model(model          = self.global_args.get('model'),
                                                       class_num      = self.global_args.get('class_num'),
                                                       in_channels    = self.global_args.get('in_channels'),
                                                       watermark_args = tmp_args)  
        return    
    
    def _regist_client(self):
        #Regist the client to the blockchain.
        for i in range(self.global_args['client_num']):
            chain_proxy.client_regist()
        self.client_list = chain_proxy.get_client_list()
    
    def _construct_client(self):
        for client_id, _ in self.client_list.items():
            # ensure each client args contains the resolved device
            client_args = dict(self.train_args) if isinstance(self.train_args, dict) else {}
            client_args['device'] = self.device
            new_client = self.client(client_id, self.train_dataloader_list[client_id], self.model,
                                     self.trainer, client_args, self.test_dataloader, self.keys_dict[client_id])
            self.client_pool.append(new_client)
    
    def run(self):
        self._regist_client()
        self._construct_dataloader()
        self._construct_sign()
        self._construct_client()
        
        for i in range(self.global_args['communication_round']):
            for client in self.client_pool:
                client.train(epoch = i)
                client.test(epoch = i)
                client.sign_test(epoch = i)
            self.server.receive_upload(self.client_pool)
            global_model = self.server.aggregate()
            for client in self.client_pool:
                client.load_state_dict(global_model)