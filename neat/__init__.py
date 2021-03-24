import logging
import sys


LOG_FORMAT = "%(asctime)s [%(threadName)-12.12s] [%(module)-10.10s] [%(levelname)-5.5s]  %(message)s"
logging.basicConfig(format=LOG_FORMAT, stream=sys.stderr, level=logging.DEBUG)