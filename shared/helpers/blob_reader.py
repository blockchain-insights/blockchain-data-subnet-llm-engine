from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.storage.blob import BlobServiceClient

import os

def download_blob_content(blob_name):
    try:
        blob_connection_string = os.environ.get("BLOB_CONNECTION_STRING")
        blob_container_name = os.environ.get("BLOB_CONTAINER_NAME",
                                     f"prompts")

        print(f"Blob connection string {blob_connection_string}")
        print(f"Blob container name: {blob_container_name}")

        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(blob_container_name)
        blob_client = container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        content = downloader.readall().decode('utf-8')

        # Close the connection
        blob_service_client.close()

        return content
        return downloader.readall().decode('utf-8')
    except Exception as e:
        print(f"Error downloading blob: {e}")
        return None