# aggregation.py

"""Aggregation on local models"""
"""We are using FedAvg in this practic coding session"""

# importing

import logging 
import time
import numpy as np
from typing import List
from .state_manager import StateManager
from fl_main.lib.util.helpers import generate_model_id
from fl_main.lib.util.states import IDPrefix


# define and initialize the aggregator class

class Aggregator : 
    """Provide a set of mathematical funcitons to compute aggregatre models"""

    def __init__(self, sm: StateManager):
        #state manager to access to models and model buffers
        self.sm = sm

    def aggregate_local_models(self):
        """Compute an average model for each tensor"""
        for mname in self.sm.mnames :
            self.sm.cluster_models[mname][0] = self.average_aggregate( self.sm.local_model_buffers[mname], self.sm.local_model_num_samples)
        
        # Save the number of samples used
        self.sm.own_cluster_num_samples = sum(self.sm.local_model_num_samples)
        logging.info(f'---cluster models are formed---')
        logging.debug(f'{self.sm.cluster_models}')

        # create model id
        id = generate_model_id(IDPrefix.aggregator, self.sm.id, time.time())
        self.sm.cluster_model_ids.append(id)

        #clear buffered local models
        self.sm.clear_lmodel_buffers()
        logging.debug('Local model buffers cleared')

    
    def  average_aggregator(self, buffer : List[np.array], num_samples: List[int]) -> np.array:
        """
        Given a list of models, compute the average model (FedAvg).
        This function provides a primitive mathematical operation.
        :param buffer: List[np.array] - A list of models to be aggregated
        :return: np.array - The aggregated models
        """

        denominator = sum(num_samples)
        # weighted average
        model = float(num_samples[0])/denominator * buffer[0]
        for i in range(1, len(buffer)):
            model += float(num_samples[i])/denominator * buffer[i]

        return model