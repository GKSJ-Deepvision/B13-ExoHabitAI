import logging

logging.basicConfig(

    filename="backend.log",

    level=logging.INFO,

    format="%(asctime)s - %(levelname)s - %(message)s"

)

logger = logging.getLogger()