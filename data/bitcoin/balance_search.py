import os

from loguru import logger
from protocols.blockchain import NETWORK_BITCOIN
from sqlalchemy import create_engine, text, func
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from settings import settings
from .balance_model import Base, BalanceChange

from data.utils.base_search import BaseBalanceSearch

class BitcoinBalanceSearch(BaseBalanceSearch):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_latest_block_number(self):
        try:
            query = select(BalanceChange).order_by(BalanceChange.block.desc()).limit(1)
            result = await self.session.execute(query)
            latest_balance_change = result.first()
            latest_block = latest_balance_change[0].block
        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            latest_block = 0
        return latest_block

    async def execute_query(self, query: str):
        # Basic check to disallow DDL queries
        ddl_keywords = ["CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE"]

        if any(keyword in query.upper() for keyword in ddl_keywords):
            raise ValueError("DDL queries are not allowed. Only data selection queries are permitted.")

        try:
            logger.info(f"Executing sql query: {query}")

            result = await self.session.execute(text(query))
            columns = result.keys()
            rows = result.fetchall()
            result = [dict(zip(columns, row)) for row in rows]
            return result

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return []

    async def execute_bitcoin_balance_challenge(self, block_height: int):
        try:
            logger.info(f"Executing balance sum query for block height: {block_height}")
            query = select(func.sum(BalanceChange.d_balance)).where(BalanceChange.block == block_height)
            result = await self.session.execute(query)
            sum_d_balance = result.scalar()
            return sum_d_balance

        except SQLAlchemyError as e:
            logger.error(f"An error occurred: {str(e)}")
            return None


    async def execute_benchmark_query(self, query: str):
        # Basic check to disallow DDL queries
        ddl_keywords = ["CREATE", "ALTER", "DROP", "TRUNCATE", "INSERT", "UPDATE", "DELETE"]

        if any(keyword in query.upper() for keyword in ddl_keywords):
            raise ValueError("DDL queries are not allowed. Only data selection queries are permitted.")

        try:
            logger.info(f"Executing sql query: {query}")

            result = await self.session.execute(text(query))
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
