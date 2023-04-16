'''
Run the MOO
'''


import MOO_B.MOO_Botorch as MOO_B
import logging
import time

if __name__=="__main__": 
    start_time = time.time()
    MOO_B.MOO_Botorch()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")