#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import logging
import shutil
from multiprocessing import Process


class InfoLogger:
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

    @staticmethod
    def change_level(new_level):
        InfoLogger.LOG_LEVEL = new_level
        logging.basicConfig(level=new_level, format=InfoLogger.LOG_FORMAT)

    @staticmethod
    def change_format(new_format):
        InfoLogger.LOG_FORMAT = new_format
        logging.basicConfig(level=InfoLogger.LOG_LEVEL, format=new_format)

    @staticmethod
    def info(message):
        logging.info(message)

    @staticmethod
    def warn(message):
        logging.warning(message)

    @staticmethod
    def debug(message):
        logging.debug(message)

    @staticmethod
    def error(message):
        logging.error(message)


class LogWriter(Process):
    def __init__(self, file_path, save_path, message_queue):
        self.name = "logging process"
        Process.__init__(self, name=self.name)
        self.file_path = file_path
        self.save_path = save_path
        self.file = None
        self.message_queue = message_queue

    def run(self) -> None:
        self.file = open(self.file_path, "a")
        self.file.truncate(0)
        while True:
            message = self.message_queue.get()
            if message == "end":
                break
            self.file.write(message + "\n")

        self.file.close()
