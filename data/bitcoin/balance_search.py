import os

from loguru import logger
from protocols.blockchain import NETWORK_BITCOIN
from sqlalchemy import create_engine, text, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker

from settings import settings
from .balance_model import Base, BalanceChange

from data.utils.base_search import BaseBalanceSearch

class BitcoinBalanceSearch(BaseBalanceSearch):
    def __init__(self, db_url: str = None):
        if db_url is None:
            self.db_url = os.environ.get("DB_CONNECTION_STRING",
                                         f"postgresql://postgres:changeit456$@localhost:5432/miner")
        else:
            self.db_url = db_url
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)

    def close(self):
        self.engine.dispose()

    def get_latest_block_number(self):
        with self.Session() as session:
            try:
                latest_balance_change = session.query(BalanceChange).order_by(BalanceChange.block.desc()).first()
                latest_block = latest_balance_change.block
            except SQLAlchemyError as e:
                logger.error(f"An error occurred: {str(e)}")
                latest_block = 0
            return latest_block

    def execute_query(self, query: str):
        # Basic check to disallow DDL queries
        ddl_keywords = ["CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE"]

        if any(keyword in query.upper() for keyword in ddl_keywords):
            raise ValueError("DDL queries are not allowed. Only data selection queries are permitted.")

        try:
            with self.Session() as session:
                logger.info(f"Executing sql query: {query}")

                result = session.execute(text(query))
                columns = result.keys()
                rows = result.fetchall()
                result = [dict(zip(columns, row)) for row in rows]
                return result

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return []

    def execute_bitcoin_balance_challenge(self, block_height: int):
        try:
            with self.Session() as session:
                logger.info(f"Executing balance sum query for block height: {block_height}")
                sum_d_balance = session.query(func.sum(BalanceChange.d_balance)).filter(BalanceChange.block == block_height).scalar()
            return sum_d_balance

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return None


    def execute_benchmark_query(self, query: str):
        # Basic check to disallow DDL queries
        ddl_keywords = ["CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE"]

        if any(keyword in query.upper() for keyword in ddl_keywords):
            raise ValueError("DDL queries are not allowed. Only data selection queries are permitted.")

        try:
            with self.Session() as session:
                logger.info(f"Executing sql query: {query}")

                result = session.execute(text(query))
                first_row = result.fetchone()
                if first_row is not None:
                    # Return the first column of the first row
                    return first_row[0]
                else:
                    # Return None if there are no rows
                    return None

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return None
