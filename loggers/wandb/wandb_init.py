from loggers.wandb.wandb_util import LoggingUtil
from rlkit.core import logger


def setup_logger(details):

	logging_tool = LoggingUtil(details)
	logger.logging_tool = logging_tool

	return logging_tool