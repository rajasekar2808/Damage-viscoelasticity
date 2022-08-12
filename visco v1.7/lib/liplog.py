#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:51:15 2021

@author: nchevaug
"""

import logging
import numpy as np


logger = logging.getLogger("liplog")

def setLogger(respath , basefilename, mode = "w+" ):  
    for handler in logger.handlers :
        handler.acquire()
        handler.flush()
        handler.close()
        handler.release()
        logger.removeHandler(handler) 
        
    if not logger.hasHandlers() :
        logger.setLevel(logging.DEBUG)
         
        # create console handler and set level to info
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        print("create logger file "+respath+"/"+basefilename+".log")
        handler = logging.FileHandler(respath+"/"+basefilename+".log",mode)
        #handler = logging.FileHandler("../tmp/Hole_damage.log",'w')
        
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class stepArraysFiles:
   
    def __init__(self, filenamebase):
             self.filenamebase = filenamebase
    def save(self, step, *args, **kwds):
        np.savez(self.filenamebase+'_%06d'%step+'.npz', *args, **kwds)
                                 
    def load(self, step):
        return np.load(self.filenamebase+'_%06d'%step+'.npz')
