import logging
import os
from typing import Optional
from azure.iot.device import IoTHubModuleClient, ProvisioningDeviceClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

# Environment variables
IOTEDGE_DEVICE_CONNECTION_STRING: Optional[str] = os.getenv('IOTEDGE_DEVICE_CONNECTION_STRING')
IOT_CENTRAL_SYMMETRIC_KEY: Optional[str] = os.getenv('IOT_CENTRAL_SYMMETRIC_KEY')
IOT_CENTRAL_ID_SCOPE: Optional[str] = os.getenv('IOT_CENTRAL_ID_SCOPE')
IOT_CENTRAL_REGISTRATION_ID: Optional[str] = os.getenv('IOT_CENTRAL_REGISTRATION_ID')

# Azure IoT Central Provisioning Host
ProvisioningHost: str = 'global.azure-devices-provisioning.net'


def provision_iot_central_device() -> str:
    """
    Provision a device in Azure IoT Central using symmetric key authentication.

    Returns:
    - str: The device connection string.
    """
    provisioning_device_client = ProvisioningDeviceClient.create_from_symmetric_key(
        symmetric_key=IOT_CENTRAL_SYMMETRIC_KEY,
        registration_id=IOT_CENTRAL_REGISTRATION_ID,
        id_scope=IOT_CENTRAL_ID_SCOPE,
        provisioning_host=ProvisioningHost
    )
    provisioning_device_client.provisioning_payload = {"a": "b"}  # Example payload

    try:
        result = provisioning_device_client.register()
        logger.info('Registration on IoT Central succeeded: %s', result)
        return f'HostName={result.registration_state.assigned_hub};DeviceId={IOT_CENTRAL_REGISTRATION_ID};SharedAccessKey={IOT_CENTRAL_SYMMETRIC_KEY}'
    except AttributeError as e:
        logger.error('Error registering device: %s', e)
        raise


def create_client() -> IoTHubModuleClient:
    """
    Create the IoT Hub client and set up the input message handler.

    Returns:
    - IoTHubModuleClient: The initialized IoT Hub client.
    """
    client = None

    try:
        conn_str = None

        if IOT_CENTRAL_REGISTRATION_ID and IOT_CENTRAL_SYMMETRIC_KEY and IOT_CENTRAL_ID_SCOPE:
            logger.info('Using IoT Central connection...')
            conn_str = provision_iot_central_device()
        elif IOTEDGE_DEVICE_CONNECTION_STRING:
            logger.info('Running in local environment using device connection string.')
            conn_str = IOTEDGE_DEVICE_CONNECTION_STRING

        if conn_str:
            client = IoTHubModuleClient.create_from_connection_string(conn_str)
        else:
            client = IoTHubModuleClient.create_from_edge_environment()

        if client is None:
            raise ValueError('Failed to create IoTHubModuleClient. The client is None.')

        logger.info('IoT Hub module client initialized')
        return client

    except Exception as e:
        logger.error(f"Could not connect: {e}", exc_info=True)
        if client:
            client.shutdown()
        raise
