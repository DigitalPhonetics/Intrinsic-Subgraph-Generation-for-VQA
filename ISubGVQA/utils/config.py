import json
from dotwiz import DotWiz
import logging


def get_config(args):
    config = json.load(open(args.config))
    config = DotWiz(config)

    logger = logging.getLogger("hnc-extension")
    logger.info(config)

    return config
