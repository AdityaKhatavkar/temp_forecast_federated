## sqlitedb.py

## this file will create the database server with the help of sqlite

#import necessaries

import logging
import time
import datetime
import sqlite3
from fl_main.lib.util.states import ModelType

class SQLiteDBHandler : 
    """ This class creates and initialize the db and inserts the model into it"""

    def __init__(self, db_file):
        self.db_file = db_file 

    def initialize_db(self):
        """This function create the db table """
        conn = sqlite3.connect(f'{self.db_file}')
        c = conn()

        #create the table for each model type

        # A)local
        c.execute('''CREATE TABLE local_models(model_id, generation_time, agent_id, round, performance, num_samples)''')

        # B) Clustor
        c.execute('''CREATE TABLE cluster_models(model_id, generation_time, aggregator_id, round, num_samples)''')

        conn.commit()
        conn.close()

    def insert_entry(self,
                         component_id: str,
                         r: int,
                         mt: ModelType,
                         model_id: str,
                         gtime: float,
                         local_prfmc: float,
                         num_samples: int
                         ):
        
        conn = sqlite3.connect(self.db_file)
        c = conn.cursor()

        t = datetime.datetime.fromtimestamp(gtime)
        gene_time = t.strftime('%m/%d/%Y %H:%M:%S')

        if mt  == ModelType.local:
            c.execute('''INSERT INTO local_models VALUES (?, ?, ?, ?, ?, ?);''', (model_id, gene_time, component_id, r, local_prfmc, num_samples))
            logging.info(f"--- Local Models are saved ---")

        elif mt == ModelType.cluster:
            c.execute('''INSERT INTO cluster_models VALUES (?, ?, ?, ?, ?);''', (model_id, gene_time, component_id, r, num_samples))
            logging.info(f"--- Cluster Models are saved ---")

        else:
            logging.info(f"--- Nothing saved ---")


        conn.commit()
        conn.close()




    