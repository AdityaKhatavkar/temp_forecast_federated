# client.py file

""" Connects locally trained ml model to the fl server and aggregator"""

"""
   Funcitons: 
    
    1) Paritcipate in FL cycle
    2) ML model ecahnge framework wiht an aggregator
    3) Push and polling mechanism to communicate with aggregator
    4) Interface between local ml application adn fl system itself
    5) provide fl client-side libraries ot ml engine 
"""

# importing necessities

import asyncio, time, logging, sys, os

from typing import Dict, Any
from threading import Thread

from fl_main.lib.util.communication_handler import init_client_server, send, receive

from fl_main.lib.util.helpers import read_config, init_loop, save_model_file, load_model_file, set_config_file, get_ip, compatible_data_dict_read, generate_model_id, create_data_dict_from_models, create_meta_data_dict, generate_id, read_state, write_state

from fl_main.lib.util.states import AggMsgType, PollingMSGLocation, ClientState, ParticipateConfirmationMSGLocation, GMDistributionMsgLocation, IDPrefix

from fl_main.lib.util.messengers import generate_lmodel_update_message, generate_agent_participation_message, generate_polling_message, generate_lmodel_update_message


# Defining the client class
"""performs participation, model exchange, communication interface bet. agent adn aggregator"""


class Client : 

    """Initialization"""
    def __init__(self):
        
        # generate unique id for client itself
        self.agent_name = "default_agent"
        self.id = generate_id()
        
        #client ip address
        self.agent_ip = get_ip()

        #is it simulation run?
        self.simulation_flag = False 
        if len(sys.argv) > 1 : 
            self.simulation_flag = bool(int(sys.argv[1]))

        # self.config file reads and stores the inofrmation of config_agent.json
        config_file = set_config_file("agent")
        self.config = read_config(config_file)

        ## ip address of the aggregator machine or instance
        self.aggr_ip = self.config['aggr_ip']
        self.reg_socket = self.config['reg_socket']  #setting up the port

        #send local ml model to the aggregaor
        self.msend_socket = 0

        # use when commmunication is not in polling mode for receiving global models sent from the aggregator
        self.exch_socket = 0
        if self.simulation_flag :
            self.exch_socket = int(sys.argv[2])
            self.agent_name = sys.argv[3]

        # stores the path ot local and global models
        self.model_path = f'{self.config["model_path"] } {self.agent_name}'
        # if there is no directory to save models
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # self.lmfile, gmfile, statefile : for sate of local modles, global models, state of client respectively
        self.lmfile = self.config['local_model_file_name']
        self.gmfile = self.config['global_model_file_name']
        self.statefile = self.config['state_file_name']

        # round information of fl process
        self.round = 0

        # Initialization    
        self.init_weights_flag = bool(self.config['init_weights_flag'])

        # Polling Method
        self.is_polling = bool(self.config['polling'])

        
# Registration of agent

    async def participate(self):

        #agent read local model to tell about its structure to aggregator
        # data_dict -> stores modes, performance_dict -> performance data
        data_dict, performance_dict  = load_model_file(self.model_path, self.lmfile)
        _, gene_time, modles, model_id = compatible_data_dict_read(data_dict)

        logging.debug(modles)

        # message 
        msg = generate_agent_participation_message(self.agent_name, self.id, model_id, modles,self.init_weights_flag, self.simulation_flag, self.exch_socket, gene_time, performance_dict, self.agent_ip)
        logging.debug(msg)
        

        # web socket generation while sending the message
        # Parse the response message
        # including some socket info and the actual round number
        resp = await send(msg, self.aggr_ip, self.reg_socket) # resp receives :  round info, port no to receive global model's exch_socket, port no. to send the local model to aggregator's msend_socket, updated agent id
        print(f"in participate() resp : {resp}")
        logging.info(f"--- Init Response: {resp} ---")
        self.round = resp[int(ParticipateConfirmationMSGLocation.round)]
        self.exch_socket = resp[int(ParticipateConfirmationMSGLocation.exch_socket)]
        self.msend_socket = resp[int(ParticipateConfirmationMSGLocation.agent_id)]

        # Receiving the welcome message
        logging.info(f'--- {resp[int(ParticipateConfirmationMSGLocation.msg_type)]} Message Received ---')

        #global model in the response is save locally by calling following funtion.
        self.save_model_from_message(resp, ParticipateConfirmationMSGLocation)

####### So, it was about the participation of the agent ###########

####### Now we will look after model exchange synchronization ###


    async def model_exchange_routine(self):
        """Check state of agent, call proper function based on the state"""

        #this process is always running while client is alive 
        """
        Check the progress of training and send the updated models
        once the training is done
        :return:
        """

       

        # Periodically check the state
        while True: 
            await asyncio.sleep(5)
            state = read_state(self.model_path, self.statefile)

            #ready to send locally trained model to aggregator
            if state == ClientState.sending: 
                await self.send_models()

            #waiting for global model           
            elif state == ClientState.waiting_gm:
                if self.is_polling == True:
                    await self.process_polling()
                else : # Do nothing
                    logging.info(f'--- Waiting for Global Model ---')   

            elif state == ClientState.training : 
                # Local model is being trained, do nothing
                logging.info(f'--- Training is happening ---')

            elif state == ClientState.gm_ready : 
                # Global model has been received, do nothing
                logging.info(f'--- Global Model is ready ---')

            else :
                logging.error(f'--- State Not Defined ---')


    #push 
    async def wait_models(self, websocket, path):
        """waiting for the cluster model from the aggregator"""
        """With the push method, aggregator will push the message that includes global models to all the connected agents right after the global models are generated"""
        gm_msg = await receive(websocket)
        logging.info(f'--- Global Model Received ---')

        logging.debug(f'Models: {gm_msg}')

        # save the global model locally
        self.save_model_from_message(gm_msg, GMDistributionMsgLocation)
        

    async def process_polling(self): 

        """with the polling method  agent keep asking the aggregator to see whether global model are already formed or not"""
        
        logging.info(f'---- polling to see if there any update')
        
        #Generate polling message to send to aggregator
        msg = generate_polling_message(self.round, self.id)

        resp = await send(msg, self.aggr_ip, self.msend_socket)

        if resp[int(PollingMSGLocation.msg_type)] == AggMsgType.update: #response msg contain updated model
            logging.info(f'--- Global Model Received ---')
            self.save_model_from_message(resp, GMDistributionMsgLocation)

        else : # AggMsgType is "ack"
            logging.info(f'--- Global Model is NOT ready (ACK) ---')

####### Fl client desing (initialization, participation,  model_exchanges) are done ##########

####### Now onwards, fl client libraries ######

####### i.e. FL funcitons

    """Now, to package essential functions to be provided as libraries to users.
In this example, the simplest way to package them as libraries will be discussed. This will need to be
expanded, depending on your needs and the design of your own FL client framework. By packaging
FL client-side modules as libraries, developers will be easily able to integrate the FL client’s functions
into the local ML engine."""


     # Starting FL client functions
    def start_fl_client(self):
        """
        Starting FL client core functions
        """
        self.register_client()
        if self.is_polling == False:
            self.start_wait_model_server()
        self.start_model_exchange_server()


    def register_client(self):
        """
        Register an agent in aggregator
        """
        time.sleep(0.5)
        asyncio.get_event_loop().run_until_complete(self.participate())

    
    def start_wait_model_server(self):
        """
        Run local ML module in parallet and receive gloval models in wait_models thread when fl system is in push communication mode"""
        time.sleep(0.5)
        th = Thread(target = init_client_server, args = [self.wait_models, self.agent_ip, self.exch_socket])
        th.start()


    def start_model_exchange_server(self):
        """
        This funciton is a thread to run model exchange routine to synchronize the local and global model while local ml module is running in parallel
        """
        time.sleep(0.05)
        self.agent_running = True
        th = Thread(target = init_loop, args =    [self.model_exchange_routine()])
        th.start()

    
    #save model from message
    def save_model_from_message(self, msg, MSG_LOC):
        # pass (model_id, models) to an app
        data_dict = create_data_dict_from_models(msg[int(MSG_LOC.model_id)], 
                        msg[int(MSG_LOC.global_models)], msg[int(MSG_LOC.aggregator_id)])
        self.round = msg[int(MSG_LOC.round)]

        # Save the received cluster global models to the local file
        save_model_file(data_dict, self.model_path, self.gmfile)
        logging.info(f'--- Global Models Saved ---')
        
        # State transition to gm_ready
        self.tran_state(ClientState.gm_ready)
        logging.info(f'--- Client State is now gm_ready ---')

    # Read and change the client state
    def read_state(self) -> ClientState:
        """
        Read the value in the state file specified by model path
        :return: ClientState - A state indicated in the file
        """
        return read_state(self.model_path, self.statefile)
    
    def tran_state(self,state: ClientState):
        """
        Change the state of the agent
        State is indicated in local file 'state'
        :param state: ClientState
        :return:
        """
        write_state(self.model_path, self.statefile, state)

    
    #send models that saved locally to the aggregator
    async def send_models(self):
        data_dict, performance_dict = load_model_file(self.model_path, self.lmfile)
        _, _, models, model_id = compatible_data_dict_read(data_dict)
        msg = generate_lmodel_update_message(self.id, model_id, models, performance_dict)

        logging.info(f'Trainded models : {msg}')

        await send(msg, self.aggr_ip, self.msend_socket)
        logging.info(f'---Local models  sent----')

        #state transition to waiting_gm
        self.tran_state(ClientState.waiting_gm)
        logging.info(f'--- Client State is now waiting_gm ---')


    def send_initial_model(self, initial_models, num_samples=1, perf_val=0.0):
        
        self.setup_sending_models(initial_models, num_samples, perf_val)
        print("send_initial_models() executed")


    def send_trained_model(self, models, num_samples, perf_values):
        state = self.read_state()
        if state == ClientState.gm_ready:
            # Do nothing: Discard the trained local models and adopt the new global models
            logging.info(f'--- The training was too slow. A new set of global models are available. ---')
        else:# Keep the training results
            # Send models
            self.setup_sending_models(models, num_samples, perf_values)



    def setup_sending_models(self, models, num_samples, perf_val):
        """Function serve as an internallibrary to set up sending locally trained modelsto the aggregator."""
        """
        Save the trained models to the local file
        :param models: np.array - models
        :param num_samples: int - Number of sample data
        :param perf_val: float - Performance data: accuracy in this case
        :return:
        """
        # Create a unique model ID
        model_id = generate_model_id(IDPrefix.agent, self.id, time.time())
        print(f"\n in setup_sending_models \n model_id => {model_id}")

        # Local Model evaluation (id, accuracy). 

        # store local ml model data
        data_dict = create_data_dict_from_models(model_id, models, self.id)
        print(f"\n in setup_sending_models \n data_dict => {data_dict}")


        # store perfrormance data
        meta_data_dict = create_meta_data_dict(perf_val, num_samples)
        print(f"\n in setup_sending_models \n meta_data_dict => {meta_data_dict}")

        save_model_file(data_dict, self.model_path, self.lmfile, meta_data_dict)
        logging.info(f'--- Local (Initial/Trained) Models saved ---')

        self.tran_state(ClientState.sending)
        logging.info(f'--- Client State is now sending ---')



    # Waiting models
    def wait_for_global_model(self):

        # Wait for global models (base models)
        print("\n wait_for_global_model function \n")
        i = 0

        while (self.read_state() != ClientState.gm_ready): #function waits until client state becomes gm_ready
            time.sleep(5)
            i = i+1
            print(f"in the loop for :{i}th time")
            

        # load models from the local file
        data_dict, _ = load_model_file(self.model_path, self.gmfile)
        global_models = data_dict['models']
        logging.info(f'--- Global Models read by Agent ---')

        self.tran_state(ClientState.training)
        logging.info(f'--- Client State is now training ---')

        return global_models


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cl = Client()
    logging.info(f'--- Your IP is {cl.agent_ip} ---')

    cl.start_fl_client()





