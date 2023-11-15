from gamechangerml.configs import PostgresConfig
from psycopg2 import connect as pg_connect


class PostgresService:
    @staticmethod
    def create_connection():
        return pg_connect(
            user=PostgresConfig.USER,
            password=PostgresConfig.PASSWORD,
            host=PostgresConfig.HOST,
            port=PostgresConfig.PORT,
            database=PostgresConfig.DATABASE,
        )

    @staticmethod
    def fetch_all(query):
        connection = PostgresService.create_connection()
        cursor = connection.cursor()
        cursor.execute(query)
        response = cursor.fetchall()
        return response

    @staticmethod
    def get_search_logs(from_date: str):
        """Get search logs from Postgres.

        Args:
            from_date (str): get logs from certain date FORMAT 'YYYY-MM-DD'

        Returns:
            Query response
        """
        query = f"SELECT * FROM gc_history WHERE run_at >= '{from_date}'::date ORDER BY run_at DESC"
        return PostgresService.fetch_all(query)
