from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.storage.blob import BlobServiceClient

def download_blob_content(connection_string, container_name, blob_name):
    try:
        blob_connection_string = os.environ.get("BLOB_CONNECTION_STRING",
                                     f"fallback")
        blob_container_name = os.environ.get("BLOB_CONTAINER_NAME",
                                     f"prompts")

        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(blob_container_name)
        blob_client = container_client.get_blob_client(blob_name)
        downloader = blob_client.download_blob()
        return downloader.readall().decode('utf-8')
    except Exception as e:
        print(f"Error downloading blob: {e}")
        return None