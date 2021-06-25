import os
import redis

from gamechangerml.api.utils.logger import logger

REDIS_HOST = os.environ.get("REDIS_HOST", default="localhost")
REDIS_PORT = os.environ.get("REDIS_PORT", default="6379")
if REDIS_HOST == "":
    REDIS_HOST = "localhost"
if REDIS_PORT == "":
    REDIS_PORT = 6379

# An easy variable interface for redis. 
# Takes in a string key and an optional boolean hash which point 
# to an index in redis and declar if it is a dictionary or not.
# Once initialized use get and set .value with and equals sign.
# Eg: latest_intel_model_sent.value = "foo"
class CacheVariable:
    def __init__(self, key, hash = False):
        self._connection = redis.Redis(connection_pool=RedisPool().getPool())
        self._key = key 
        self._hash = hash
    def get_value(self):
        if(self._connection.exists(self._key)):
            if(self._hash):
                return self._connection.hgetall(self._key)
            return self._connection.get(self._key)
        return None
    def set_value(self, value):
        if(self._hash):
            return self._connection.hmset(self._key, value)
        return self._connection.set(self._key, value)
    def del_value(self):
        return self._connection.delete(self._key)
    
    value = property(get_value, set_value, del_value)

# A singleton class that creates a connection pool with redis.
# All cache variables use this one connection pool.
class RedisPool:
    __pool = None
    @staticmethod 
    def getPool():
        """ Static access method. """
        if RedisPool.__pool == None:
            RedisPool()
        return RedisPool.__pool
    def __init__(self):
        """ Virtually private constructor. """
        if RedisPool.__pool != None:
            logger.info("Using redis pool singleton")
        else:
            try:
                RedisPool.__pool = redis.ConnectionPool(host=REDIS_HOST, port=int(REDIS_PORT), db = 0)
            except Exception as e:
                logger.error(
                    " *** Unable to connect to redis {REDIS_HOST} {REDIS_PORT}***")
                logger.error(e)