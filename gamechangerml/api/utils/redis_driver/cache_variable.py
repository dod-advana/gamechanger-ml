from json import loads, dumps
from redis import Redis
from .redis_pool import RedisPool


REDIS_CONNECTION = Redis(connection_pool=RedisPool().getPool())


class CacheVariable:
    def __init__(self, key, encode=False):
        self._connection = REDIS_CONNECTION
        self._key = key
        self._encode = encode
        self.test_value = None

    # Default get method, checks if the key is in redis and gets
    # the value whether it is a list, dict or standard type
    def get_value(self):
        try:
            if self._connection.exists(self._key):
                result = self._connection.get(self._key)
                if self._encode:
                    result = loads(result)
                return result
            return None
        except Exception as e:
            print(e)
            return self.test_value

    # Default set method, sets values for dicts and standard types.
    # Note: Should use push if using a list.
    def set_value(self, value, expire=None):
        try:
            if self._encode:
                value = dumps(value)
            if expire:
                self._connection.set(self._key, value)
                self._connection.expireat(self._key, expire)
            else:
                self._connection.set(self._key, value)
        except Exception as e:
            print(e)
            self.test_value = value

    # Default delete method, removes key from redis
    def del_value(self):
        return self._connection.delete(self._key)

    value = property(get_value, set_value, del_value)
