import logging
from typing import Dict, Any
from datetime import datetime
import json
import uuid
import pandas as pd
import numpy as np
from azure.iot.device import Message
from azure.iot.device import IoTHubModuleClient

logger = logging.getLogger(__name__)


def default_serializer(obj):
    """
    Helper function to convert non-serializable objects to JSON-serializable format.

    Parameters:
    - obj: The object to serialize.

    Returns:
    - A JSON-serializable version of the object.
    """

    if isinstance(obj, (pd.Timestamp, np.datetime64, datetime)):
        # Convert Timestamp and datetime objects to ISO format strings
        return obj.isoformat()

    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)

    if isinstance(obj, datetime):
        return obj.isoformat()

    return obj


def serialize_for_json(obj):
    """
    Recursively checks every property value of an object and replaces values
    that cannot be serialized by JSON.

    Parameters:
    - obj: The input object to serialize.

    Returns:
    - A JSON-serializable object.
    """
    if isinstance(obj, dict):
        # If the object is a dictionary, process each key-value pair recursively
        return {k: serialize_for_json(v) for k, v in obj.items()}

    if isinstance(obj, list):
        # If the object is a list, process each element recursively
        return [serialize_for_json(v) for v in obj]

    if isinstance(obj, tuple):
        # If the object is a tuple, convert it to a list and process each element
        return tuple(serialize_for_json(v) for v in obj)

    if isinstance(obj, (pd.Timestamp, np.datetime64, datetime)):
        # Convert Timestamp and datetime objects to ISO format strings
        return obj.isoformat()

    if isinstance(obj, (np.int64, np.int32)):
        # Convert numpy int64 or int32 to Python int
        return int(obj)

    if isinstance(obj, (np.float64, np.float32)):
        # Convert numpy float64 or float32 to Python float
        return float(obj)

    if isinstance(obj, np.generic):
        # Convert other numpy generic types to their native Python equivalents
        return obj.item()

    if isinstance(obj, set):
        # Convert sets to lists (JSON does not support sets)
        return list(obj)

    # For all other types that are natively JSON serializable, return the object as is
    return obj


def send_to_hub(client: IoTHubModuleClient, data: Dict[str, Any] = None, properties=None,
                output_name: str = 'output1') -> None:
    """
    Send data to Azure IoT Hub using IoT Edge messages.

    Parameters:
    - client: IoTHubModuleClient
        The existing IoT Hub client to use for sending messages.
    - data: Dict[str, Any]
        The data to send to Azure IoT Hub.
    - output_name: str
        The output route name defined in the IoT Edge deployment manifest.
    """
    try:
        if properties:
            properties = serialize_for_json(properties)
            client.patch_twin_reported_properties(properties)
        else:
            if data is None:
                data = {}

            payload = json.dumps(data, default=default_serializer)
            message = Message(payload)
            message.message_id = uuid.uuid4()

            client.send_message_to_output(message, output_name)

        logger.info(f"Sent data to Azure IoT Hub on output '{output_name}': {data}")
    except Exception as e:
        logger.error(f"Failed to send data to IoT Hub on output '{output_name}': {e}", exc_info=True)
        raise

