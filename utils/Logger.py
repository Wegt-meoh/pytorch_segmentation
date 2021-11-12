import logging
import os
import sys

def get_logger(description,save_dir,filename='log.txt',mode='a+'):
    logger=logging.getLogger(description)
    logger.setLevel(logging.DEBUG)
    ch=logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        fh=logging.FileHandler(os.path.join(save_dir,filename),mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger        
