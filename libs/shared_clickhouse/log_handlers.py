import json
import logging
from datetime import datetime

import clickhouse_connect

from .db import ClickHouseDatabase, initialize_logs_table


class ClickHouseLogHandler(logging.Handler):
    def __init__(self, db: ClickHouseDatabase, service_name: str, table="logs"):
        """
        Initialize the ClickHouse log handler.
        """
        super().__init__()
        self.client = db.get_client()

        initialize_logs_table(db, table)
        self.table = table

        self.service_name = service_name

    def emit(self, record):
        try:
            # log_entry = {
            #     'timestamp': datetime.fromtimestamp(record.created),
            #     'service_name': self.service_name,
            #     'logger_name': record.name,
            #     'log_level': record.levelname,
            #     'message': record.getMessage(),
            #     'extra_fields': json.dumps(record.__dict__.get('extra', {}))
            # }
            log_entry = [
                datetime.fromtimestamp(record.created),
                self.service_name,
                record.name,
                record.levelname,
                record.getMessage(),
                json.dumps(record.__dict__.get("extra", {})),
            ]
            self.client.insert(self.table, [log_entry])
        except Exception:
            self.handleError(record)
