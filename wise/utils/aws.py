import boto3

def upload_file_to_s3(file_path: str, bucket_name: str, object_key: str) -> None:
    """
    Uploads a file from the local file system to an S3 bucket.

    Parameters
    ----------
    file_path : str
        Path to the local file.
    bucket_name : str
        Name of the S3 bucket.
    object_key : str
        Key (path/filename) to use for the uploaded file in S3.
    """
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket_name, object_key)
    print(f"File {file_path} uploaded to s3://{bucket_name}/{object_key}")

def read_object_from_s3(bucket_name: str, object_key: str) -> bytes:
    """
    Read an object from an S3 bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    object_key : str
        Key (path/filename) of the object in S3.

    Returns
    -------
    bytes
        The content of the object as bytes.
    """
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    content = response['Body'].read()
    print(f"Successfully retrieved object s3://{bucket_name}/{object_key}")
    return content

def create_folder_if_not_exists(bucket_name: str, folder_name: str) -> None:
    """
    Create a folder in an S3 bucket if it does not already exist.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    folder_name : str
        Name of the folder to create (should end with '/').
    """
    s3 = boto3.client('s3')
    # Create a zero-byte object with the folder name
    s3.put_object(Bucket=bucket_name, Key=folder_name)
    print(f"Folder {folder_name} created in bucket {bucket_name}")

def get_latest_nc_file_from_s3(bucket_name: str, prefix: str) -> str:
    """
    Recursively searches for the most recently modified .nc file under the given prefix in the S3 bucket.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket.
    prefix : str
        Folder path (prefix) in the S3 bucket to search in.

    Returns
    -------
    str
        The key of the latest .nc file found. If no file is found, returns an empty string.
    """
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    nc_objects = []
    for page in page_iterator:
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".nc"):
                    nc_objects.append(obj)

    if not nc_objects:
        print(f"No .nc files found under {prefix} in bucket {bucket_name}")
        return ""

    latest_obj = max(nc_objects, key=lambda o: o["LastModified"])
    print(f"Latest .nc file found: s3://{bucket_name}/{latest_obj['Key']}")
    return latest_obj["Key"]