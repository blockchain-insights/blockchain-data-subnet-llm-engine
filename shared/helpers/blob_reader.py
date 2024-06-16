from azure.storage.blob import BlobServiceClient
from loguru import logger
import os


def download_blob_content(blob_name):
    try:
        blob_connection_string = os.environ.get("BLOB_CONNECTION_STRING")
        blob_container_name = os.environ.get("BLOB_CONTAINER_NAME")
        if blob_connection_string is None:
            return None
        if blob_container_name is None:
            return None

        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(blob_container_name)
        blob_client = container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        content = downloader.readall().decode('utf-8')

        blob_service_client.close()

        return content
    except Exception as e:
        logger.error(f"Error downloading blob: {e}")
        return None
