from os import environ

class PostgresConfig:
    USER = "postgres"
    PASSWORD = "password"
    HOST = environ.get("PG_HOST", "localhost")
    PORT = 5432 
    DATABASE = "uot"
