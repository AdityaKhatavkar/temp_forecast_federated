# pseudo_db.py

""" Acepting  messages that contain local and cluster global models """

# import essentials 

import pickle, logging, time, os
from typing import Any, List
from .sqlitedb import SQLiteDBHandler
from fl_main.lib.util.helpers import generate_id, read_config, set_config_file
from fl_main.lib.util.states import DBMsgType, DBPushMsgLocation, ModelType
from fl_main.lib.util.communication_handler import init_db_server, send_websocket, receive


class PseudoDB : 
    """
    this class's instance receives models and their data from aggregator 
    and pushes them to the actual  database"""

    # initializing pseudoDB

    def __init__(self):
        

            # Database ID just in case
            self.id = generate_id()

            # read the config file
            self.config = read_config(set_config_file("db"))
            

            # Initialize DB IP and Port
            self.db_ip = self.config['db_ip']
            self.db_socket = self.config['db_socket']

            # if there is no directory to save models create the dir
            self.data_path = self.config['db_data_path']
            if not os.path.exists(self.data_path) : 
                os.makedirs(self.data_path)

            # int DB
            self.db_file = f'{self.data_path}/model_data{time.time()}.db'
            self.dbhandler =  SQLiteDBHandler(self.db_file)
            self.dbhandler.initialize_db()

            # Model save location
            # if there is no directory to save models
            self.db_model_path = self.config['db_model_path']
            if not os.path.exists(self.db_model_path):
                os.makedirs(self.db_model_path)


    async def handler(self, websocket, path):
        
        """Takes websocket as a parameter, receivermessages from the aggregator and returns the requested information"""
        
        msg = await receive(websocket) # receive message from the aggregator
        logging.info(f'Request Arrived')
        logging.debug(f'Request: {msg}')

        msg_type = msg[DBPushMsgLocation.msg_type] # decode the msg type
        reply = list()

        if msg_type == DBMsgType.push: # if msg type is push then 
            logging.info(f'--- Model pushed: {msg[int(DBPushMsgLocation.model_type)]} ---')
            self.push_all_data_to_db(msg) # it will push the local/cluster models to the database
            reply.append('confirmation')
        else: #otherwise
            raise TypeError(f'Undefined DB Message Type : {msg_type}') # it will show an error message
        
        # Reply to the aggregator
        await send_websocket(reply, websocket)

    
    def push_all_data_to_db(self, msg : List[Any]):

        """push all the models info the the datbase"""
        """
        push data received from the aggregator to database 
        and save models in the file system
        :param msg: Message received
        :return: component id, round, message typr, model id, gene time, local perf, num samples
        """

        pm = self.parse_message(msg) # extract message content
        self.dbhandler.insert_entry(*pm) 

        #save model
        model_id = msg[int(DBPushMsgLocation.model_id)]
        models = msg[int(DBPushMsgLocation.models)]
        fname = f'{self.db_model_path}/{model_id}.binaryfile'
        with open(fname, 'wb') as f :
            pickle.dump(models, f)


    def parse_message(self, msg : List[Any]):
        component_id = msg[int(DBPushMsgLocation.component_id)]
        r = msg[int(DBPushMsgLocation.round)]
        mt = msg[int(DBPushMsgLocation.model_type)]
        model_id = msg[int(DBPushMsgLocation.model_id)]
        gene_time = msg[int(DBPushMsgLocation.gene_time)]
        meta_data = msg[int(DBPushMsgLocation.meta_data)]

        # if local model performance is saved
        local_prfmc = 0.0
        if mt == ModelType.local:
            try : local_prfmc = meta_data["accuracy"]
            except:  pass

        # Number of samples is saved
        num_samples = 0
        try : num_samples = meta_data["num_samples"]
        except : pass


        return component_id, r, mt, model_id, gene_time, local_prfmc, num_samples

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("--- Pseudo DB Started ---")

    pdb = PseudoDB()
    init_db_server(pdb.handler, pdb.db_ip, pdb.db_socket)