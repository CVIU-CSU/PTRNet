#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File        :config.py
@Description :
@Time        :2021/04/12 09:17:49
@Author      :Jinkui Hao
@Version     :1.0
'''
import sys
import os


class LoggerMST(object):
    def __init__(self, train_pth, test_pth):
        self.terminal = sys.stdout
        
        self.train_file = open(train_pth, "a")
        self.test_file = open(test_pth, "a")
        sys.stdout = self

    def write(self, message, mode='train'):
        self.terminal.write(message)
        if mode=='test':
            self.train_file.write(message)
        else:
            self.test_file.write(message)

    def flush(self):
        self.terminal.flush()
        self.train_file.flush()
        self.test_file.flush()
        

class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout

        if not os.path.exists(outfile):
            os.system(r"touch {}".format(outfile))
        
        self.log = open(outfile, "a")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()
        

    


   
    
    