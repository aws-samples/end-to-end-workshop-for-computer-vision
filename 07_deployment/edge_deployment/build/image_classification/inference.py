import grpc
from PIL import Image
import agent_pb2 as agent
import agent_pb2_grpc as agent_grpc
import os
import time
import numpy as np    
from numpy import argmax
import uuid

# This is a test script to check if my stuff are uploaded properly
model_path = '/greengrass/v2/work/Bird-Model-ARM-TF2'
model_name = 'bird-model'

image_path = os.path.expandvars(os.environ.get("DEFAULT_SMEM_IC_IMAGE_DIR"))
                    
agent_socket = 'unix:///tmp/aws.greengrass.SageMakerEdgeManager.sock'

channel = grpc.insecure_channel(agent_socket)
client = agent_grpc.AgentStub(channel)

print(f"Images directory is {image_path}================")

print(f"our current directory is {os.getcwd()}==========")


def list_models(cli):
    resp = cli.ListModels(agent.ListModelsRequest())
    return {
        m.name: {"in": m.input_tensor_metadatas, "out": m.output_tensor_metadatas}
        for m in resp.models
    }


def unload_model(cli, model_name):
    try:
        req = agent.UnLoadModelRequest()
        req.name = model_name
        return cli.UnLoadModel(req)
    except Exception as e:
        print(e)
        return None


def load_model(cli, model_name, model_path):
    """ Load a new model into the Edge Agent if not loaded yet"""
    try:
        req = agent.LoadModelRequest()
        req.url = model_path
        req.name = model_name
        return cli.LoadModel(req)
    except Exception as e:
        print(e)
        return None


def create_tensor(x, tensor_name):
    if x.dtype != np.float32:
        raise Exception("It only supports numpy float32 arrays for this tensor")
    tensor = agent.Tensor()
    tensor.tensor_metadata.name = tensor_name
    tensor.tensor_metadata.data_type = agent.FLOAT32
    for s in x.shape:
        tensor.tensor_metadata.shape.append(s)
    tensor.byte_data = x.tobytes()
    return tensor


def predict(cli, model_name, x, shm=False):
    """
    Invokes the model and get the predictions
    """
    model_map = list_models(cli)
    if model_map.get(model_name) is None:
        raise Exception("Model %s not loaded" % model_name)
    # Create a request
    req = agent.PredictRequest()
    req.name = model_name
    # Then load the data into a temp Tensor
    tensor = agent.Tensor()
    meta = model_map[model_name]["in"][0]
    tensor.tensor_metadata.name = meta.name
    tensor.tensor_metadata.data_type = meta.data_type
    for s in meta.shape:
        tensor.tensor_metadata.shape.append(s)

    if shm:
        tensor.shared_memory_handle.offset = 0
        tensor.shared_memory_handle.segment_id = x
    else:
        tensor.byte_data = x.astype(np.float32).tobytes()

    req.tensors.append(tensor)

    # Invoke the model
    resp = cli.Predict(req)

    # Parse the output
    meta = model_map[model_name]["out"][0]
    tensor = resp.tensors[0]
    data = np.frombuffer(tensor.byte_data, dtype=np.float32)
    return data.reshape(tensor.tensor_metadata.shape)


def main():
    if list_models(client).get(model_name) is None:
        load_model(client, model_name, model_path)

    print("Loaded Models:")
    print(list_models(client))

    start = time.time()
    
    count = 0
    
    if os.path.isdir(image_path):
        for images in os.listdir(image_path):
            if ".jpg" in images:
                im = Image.open(f"{image_path}/{images}")
                input_image = np.array(im).astype(np.float32) / 255.0
                print(f"Shape of the Image {input_image.shape}============")

                y = predict(client, model_name, input_image, False)
                
                print(f"output tensor is {y}===================")
                predicted_class_idx = argmax(y)

                total_time = time.time() - start
                print(f"Output Prediction: {predicted_class_idx}")

                req = agent.CaptureDataRequest()
                req.model_name = model_name
                req.capture_id = str(uuid.uuid4())
                req.input_tensors.append(create_tensor(input_image, "input_image"))
                req.output_tensors.append(create_tensor(y, "output_image"))
                resp = client.CaptureData(req)
                
                count +=1
    else:
        print("Images directory not found ===============")
        
    print(f"Number of images predicted: {count}============")
        
    print("End of the script ==============")


if __name__ == '__main__':
    main()