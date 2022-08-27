#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:51:15 2021

@author: nchevaug
"""

import logging
import numpy as np
import time


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
    
    
class timer:
    def __init__(self):
        self.clocks = {}
    def new_increment(self):
        for k  in self.clocks.keys() :
          self.clocks[k].update({'icumt':0., 'icumtp':0.} )
          
    def start(self, clockname):
        tmp = self.clocks.get(clockname)
        if tmp is None :
            self.clocks[clockname]={'icumt':0., 'cumt':0.,'icumtp':0.,'cumtp':0.} 
        self.clocks[clockname].update({'t0':time.time(), 'tp0':time.process_time()})
    def end(self, clockname):
        tmp = self.clocks.get(clockname)
        if tmp is None : raise
        deltat = time.time() - tmp['t0']
        deltatp = time.process_time() - tmp['tp0']
        self.clocks[clockname].update({'cumt':tmp['cumt']+ deltat })
        self.clocks[clockname].update({'cumtp':tmp['cumtp']+ deltatp })
        self.clocks[clockname].update({'icumt':tmp['icumt']+ deltat })
        self.clocks[clockname].update({'icumtp':tmp['icumtp']+ deltatp })
        
    def log(self, logger = logger):
        
        logger.info('{:10s} {:10s}  {:10s} {:10s}  {:10s}'.format('name', 'inc time(s)', 'inc cpu time(s)', 'tot time(s)', 'tot cpu time(s)'))
        for k, v  in self.clocks.items() :
            logger.info('{:10s} {:10.2f}  {:16.2f} {:11.2f}  {:15.2f}'.format(k, v['icumt'], v['icumtp'], v['cumt'], v['cumtp']))

