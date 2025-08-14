# server_.py

'''
What this file does ?
1) communication processes bet. aggregator itself, agents,  database to coordinating agent participation,  aggregation of ml models.
2) receiving local models
3) cluster model synthesis routine 
'''

# importing  the  necessities


# creating class
    #functionalities of class 
'''         
            1) agent registration
            2) global model synthesis
            3) handeling mechanisms of uploaded local models and polling messages sent from agents
            4) interface bet. aggregator and database
            5) interface bet. aggregator and agents
'''


class server : 
    
    def __init__(self):
        """
        Instantiation of a Server instance
        """

        # read the config file
        config_file = set_config_file("aggregator") 
        self.config = read_config(config_file) #stores info from congig_aggregator.json file

        # functional components
        self.sm = statemanager()
        self.agg = aggregator(self.sm) #aggregation functions

        # Set up FL server's IP address
        self.aggr_ip = self.config['aggr_ip'] #reads ip addr fron aggregator's configuration file

        # port numbers, websocket info
        self.reg_socket = self.config['reg_socket'] #reg_socket is used for agents toregister themselves 
        self.recv_socket = self.config['recv_socket'] #recv_socket used to reciv local models from agents
        self.exch_socket = self.config['exch_socket']#exch_socket is port no.used to send the global model back to the agent

        # Set up DB info to connect with DB
        self.db_ip = self.config['dp_ip'] #ip addr of database server
        self.db_socket = self.config['db_socket'] #port no of database server

        # thresholds
        self.round_interval = self.config['round_interval']
        self.is_polling = bool(self.config['polling'])

        self.sm.agg_threshold = self.config['aggregation_threshold']





#Asynchronous programming is a way of writing code so that tasks that take time (like network calls, file I/O, API requests, etc.) don’t block the rest of your program.

#coroutines (functions declared with async def)

#async def : When you call it, it does not run immediately — it returns a coroutine object. You need to run that coroutine using await or an event loop:

    async def register(self, websocket: str, path):
        """
        Receiving the participation message specifying the model structures
        Sending back socket information for future model exchanges.
        Sending back the welcome message as a response.
        :param websocket:
        :param path:
        :return:
        """
        # Receiving participation messages
        msg = await receive(websocket)  # receive msg from agent and store in 'msg'
        logging.info(f'--- {msg[int(ParticipateMSGLocation.msg_type)]} Message Received ---')
        logging.debug(f'Message: {msg}')

        # Check if it is a simulation run
        es = self._get_exch_socket(msg)

        # Add an agent to the agent list
        agent_name = msg[int(ParticipateMSGLocation.agent_name)]
        agent_id = msg[int(ParticipateMSGLocation.agent_id)]
        addr = msg[int(ParticipateMSGLocation.agent_ip)]

        uid, ues = self.sm.add_agent(agent_name, agent_id, addr, es)

        # If the weights in the first models should be used as the init models
        # The very first agent connecting to the aggregator decides the shape of the models
        if self.sm.round == 0:
            await self._initialize_fl(msg)

        # If there was at least one global model, just proceed

        # Wait for sending messages
        await asyncio.sleep(0.5)

        # send back 'welcome' message
        await self._send_updated_global_model(websocket, uid, ues)




    def _get_exch_socket(self, msg):

        """The server has multiple sockets (ports) for different purposes — registration, receiving models, and exchanging models back.
        _get_exch_socket() decides which port number to use when sending the global model back to a specific agent:
        Simulation run → use the port specified by that agent in its message.
        Normal run → use the server's default exchange socket."""

        if msg[int(ParticipateMSGLocation.sim_flag)]: #check it this is simulation run
            logging.info("--This run is sumulation")
            es = msg[int(ParticipateMSGLocation.exch_socket)]
        else : 
            es = self.exch_socket
        return es
    
        """If it is simulation run, then u can run u can run all fl componnents (database, agents, aggregator) on one machine"""



    async def _initialize_fl(self, msg):
        """Initialize the fl process, only call when fl round is 0. Push local model to db by calling the funciton _push_local_models()"""
        """
        Initialize FL round
        :param msg: Message received
        :return:
        """
        # Extract values from the message received
        agent_id = msg[int(ParticipateMSGLocation.agent_id)]
        model_id = msg[int(ParticipateMSGLocation.model_id)]
        gene_time = msg[int(ParticipateMSGLocation.gene_time)]
        lmodels = msg[int(ParticipateMSGLocation.lmodels)] # <- Extract local models
        performance = msg[int(ParticipateMSGLocation.meta_data)]
        init_weight_flag = bool(msg[init(ParticipateMSGLocation.init_flage)])

        # Initialize model info 
        self.sm.initialize_model_info(lmodels, init_weight_flag)

        # pushign local model to DB
        await self._push_local_models(agent_id, model_id, lmodels,  gene_time, performance)

        self.sm.increment_round()



    async def _send_updated_global_model(self, websocket, agent_id, exch_socket):

        model_id = self.sm.cluster_model_ids[-1]
        cluster_models = convert_Ldict_to_dict(self.sm.cluster_models)

        reply = generate_agent_participation_confirm_message(self.sm.id, model_id, cluster_models, self.sm.round, agent_id, exch_socket, self.recv_socket)
        await send_websocket(reply, websocket)



############################# upto this we saw the agent registration process ############################
############# now onwards we will see the handling the local ml models and polling messages ##############



    async def receive_msg_from_agent(self, websocket, path):
        """This process is constantly running to receive local model updates from the agent and push them to database """
        """
        Receiving messages from agents for model updates or polling
        :param websocket:
        :param path:
        :return:
        """
        msg = await receive(websocket)
        if msg[int(ModelUpMSGLocation.msg_type)] == AgentMSGType.update : 
            await self._process_lmodel_upload(msg)

        elif msg[int(PollingMSGLocation.msg_type)] == AgentMSGType.polling : 
            await self._process_polling(msg, websocket)

            


    async def _process_lmodel_upload(self, msg):
        "receive local ml model from agent and push them to the buffer"
        """
        Process local models uploaded from agents
        :param msg: message received from the agent
        :return:
        """
        lmodels = msg[int(ModelUpMSGLocation.lmodels)]
        agent_id = msg[int(ModelUpMSGLocation.agent_id)]
        model_id = msg[int(ModelUpMSGLocation.model_id)]
        gene_time = msg[int(ModelUpMSGLocation.gene_time)]
        perf_val = msg[int(ModelUpMSGLocation.meta_data)]
        await self._push_local_models(agent_id, model_id, lmodels, gene_time, perf_val) #push local model to database

        logging.info('--- Local Model Received ---')
        logging.debug(f'Local models: {lmodels}')

        # Store local models in the buffer
        self.sm.buffer_local_models(lmodels, participate=False, meta_data=perf_val)



    async def _process_polling(self, msg, websocket):
        logging.debug(f'--- AgentMsgType.polling ---')

        if self.sm.round > int(msg[PollingMSGLocation.round]):
            model_id = self.sm.cluster_model_ids[-1]
            cluster_models = convert_Ldict_to_dict(self.sm.cluster_models)
            msg = generate_cluster_model_dict_message(self.sm.id, model_id, self.sm.round, cluster_models)
        
        else : 
            ack = generate_ack_message()
            await send_websocket(msg, websocket)




    async def model_synthesis_routine(self):
        """periodically cahecking stored models, perform global model synthesis if no. of models > threshold """

        while True: 
            # Periodic check (frequency is specified in the JSON config file)
            await asyncio.sleep(self.round_interval)

            if self.sm.ready_for_local_aggregation() : # <- if it has enough models to aggregate 
                logging.info(f'Round {self.sm.round}')
                logging.info(f'Current agents: {self.sm.agent_set}')

                # ----- Local aggregation process -----#
                # local models ---> an cluster model
                # create a cluster model from local models 
                self.agg.aggregate_local_models()

                # push cluster model to db
                await self.push_cluster_models()
                
                if self.is_polling() == False: #send global model to all connected agents if polling method is not used
                    await self.send_cluster_models_to_all() 

                self.sm.increment_round()



    async def send_cluster_models_to_all(self):
        # send out cluster models to all agents under this aggregator
        model_id = self.sm.cluster_model_ids[-1]
        cluster_models = convert_LDict_to_Dict(self.sm.cluster_models)
        msg = generate_cluster_model_dist_message(self.sm.id, model_id, self.sm.round, cluster_models)

        for agent in self.sm.agent_set:
            await send(msg, agent['agent_ip'], agent['socket'])



######## So, about sending cluster global model to all agent is done. Now w'll look after local & cluster models to the database
    async def _push_local_models(self, agent_id: str, model_id: str, local_models: Dict[str, np.array],\
                                 gene_time: float, performance: Dict[str, float]) -> List[Any]:
        
        #Pushing a given set of local models to DB
        """
        :param agent_id: str - ID of the agent that created this local model
        :param model_id: str - Model ID passed from the agent
        :param local_models: Dict[str,np.array] - Local models
        :param gene_time: float - the time at which the models were generated
        :param performance: Dict[str,float] - Each entry is a pair of model ID and its performance metric
        :return: Response message (List)
        """
        logging.debug(f'The local models to send: {local_models}')
        return await self._push_models(agent_id, ModelType.local, local_models, model_id, gene_time, performance)
    


    async def _push_cluster_models(self) -> List[Any]:
        """Pushing cluster models to DB"""




