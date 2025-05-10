import clickhouse_connect


class ClickHouseDatabase:
    def __init__(self, host: str, port: str, user: str, password: str, database: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database

    def get_client(self):
        """
        Get a ClickHouse client.
        """
        return clickhouse_connect.get_client(
            host=self.host,
            port=self.port,
            username=self.user,
            password=self.password,
            database=self.database,
        )


def initialize_logs_table(db: ClickHouseDatabase, table_name: str):
    """
    Initialize the given table as a table compatible with logging, in a ClickHouse database if it doesn't exist.
    """
    client = db.get_client()
    client.command(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            timestamp DateTime,
            service_name String,
            logger_name String,
            log_level String,
            message String,
            extra_fields String
        ) ENGINE = MergeTree()
        ORDER BY (timestamp, service_name)
        """
    )
