from redis import ConnectionPool
from gamechangerml.api.utils.logger import logger
from gamechangerml.configs import RedisConfig


class RedisPool:
    """A singleton class that creates a connection pool with redis. All cache
    variables use this one connection pool.
    """

    __pool = None

    @staticmethod
    def getPool():
        """Static access method."""
        if RedisPool.__pool == None:
            RedisPool()
        return RedisPool.__pool

    def __init__(self):
        """Virtually private constructor."""
        if RedisPool.__pool != None:
            logger.info("Using redis pool singleton")
        else:
            try:

                RedisPool.__pool = ConnectionPool(
                    host=RedisConfig.HOST,
                    port=int(RedisConfig.PORT),
                    db=0,
                    decode_responses=True,
                )
            except Exception as e:
                logger.error(
                    f"*** Unable to connect to redis {RedisConfig.HOST} {RedisConfig.PORT}***"
                )
                logger.error(e)
