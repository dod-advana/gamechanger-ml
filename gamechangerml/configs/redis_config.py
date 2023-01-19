from os import environ

class RedisConfig:
    HOST = environ.get("REDIS_HOST", default="localhost")
    PORT = environ.get("REDIS_PORT", default=6379)
