import os
import json
import boto3
import requests
import numpy as np
from io import BytesIO

def get_s3_client():

    session = boto3.Session(profile_name=os.environ["AWS_PROFILE"])
    return session.client('s3')

def load_reps_from_aws(file_path: str,
                       train_test: str):

    try:
        aws_file_path = file_path.replace(f"{os.environ['DATA_SOURCE']}/", '')
        client = get_s3_client()
        response = client.get_object(Bucket=os.environ["AWS_BUCKET"], Key=aws_file_path)
        with BytesIO(response['Body'].read()) as fd:
            fd.seek(0)
            loader = np.load(fd, allow_pickle=True)
            return { 
                f"X_{train_test}": loader[f"X_{train_test}"],
                f"y_{train_test}": loader[f"y_{train_test}"]
            }
    except Exception as e:
        print("Error:", e)

def store_nparrays_in_aws(file_path: str,
                          dict_arrays: dict):
    try:
        aws_file_path = file_path.replace(f"{os.environ['DATA_SOURCE']}/", '')
        with BytesIO() as fd:
            np.savez(fd, **dict_arrays)
            client = get_s3_client()
            fd.seek(0)
            _ = client.put_object(Bucket=os.environ["AWS_BUCKET"],
                                Key=aws_file_path,
                                Body=fd.read())
    except Exception as e:
        print("Error: ", e)
    
def store_json_in_aws(file_path: str,
                      json_dict: str):
    try:
        aws_file_path = file_path.replace(f"{os.environ['DATA_SOURCE']}/", '')
        client = get_s3_client()
        client.put_object(Bucket=os.environ["AWS_BUCKET"],
                          Key=aws_file_path,
                          Body=json.dumps(json_dict).encode())
    except Exception as e:
        print("Error: ", e)


def load_json_from_aws(file_path: str):
    
    try:
        aws_file_path = file_path.replace(f"{os.environ['DATA_SOURCE']}/", '')
        client = get_s3_client()
        response = client.get_object(Bucket=os.environ["AWS_BUCKET"], Key=aws_file_path)
        data = response["Body"].read()
        return json.loads(data.decode())
    except Exception as e:
        print("Error: ", e)

def aws_path_exists(file_path: str):
    try:
        aws_file_path = file_path.replace(f"{os.environ['DATA_SOURCE']}/", '')
        client = get_s3_client()
        client.head_object(Bucket=os.environ["AWS_BUCKET"], Key=aws_file_path)
        return True  # File exists
    except:
        return False  # File doesn't exist

def aws_stop_instance():

    # Getting the permission token.
    url = "http://169.254.169.254/latest/api/token"
    headers = {"X-aws-ec2-metadata-token-ttl-seconds": "21600"}
    response = requests.put(url, headers=headers)
    if response.status_code == 200:
        token = response.text
        print(f"Token: {token}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    # Getting the instance ID.
    url = "http://169.254.169.254/latest/meta-data/instance-id"
    headers = {"X-aws-ec2-metadata-token": token}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        instance_id = response.text
        print(f"Instance ID: {instance_id}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

    ec2 = boto3.client('ec2')
    ec2.stop_instances(InstanceIds=[instance_id])