'''
Run the MOO
'''


import MOO.affinity_MOO as MOO
import logging
import time

if __name__=="__main__": 
    start_time = time.time()
    MOO.MOO()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")