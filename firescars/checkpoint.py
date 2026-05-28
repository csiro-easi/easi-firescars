"""Checkpoint save/load to local or S3."""

import io
import os

import torch


def _is_s3(path):
    return path.startswith("s3://")


def save_checkpoint(state, path, is_best=False):
    """Save checkpoint to local path or S3."""
    if _is_s3(path):
        import boto3

        s3 = boto3.client("s3")
        bucket, key = path[5:].split("/", 1)

        buf = io.BytesIO()
        torch.save(state, buf)
        buf.seek(0)
        s3.put_object(Bucket=bucket, Key=key + "model_latest.pth", Body=buf.getvalue())

        if is_best:
            buf.seek(0)
            s3.put_object(Bucket=bucket, Key=key + "model_best.pth", Body=buf.getvalue())
    else:
        os.makedirs(path, exist_ok=True)
        torch.save(state, os.path.join(path, "model_latest.pth"))
        if is_best:
            torch.save(state, os.path.join(path, "model_best.pth"))


def load_checkpoint(path):
    """Load checkpoint from local path or S3. Returns None if not found."""
    if _is_s3(path):
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")
        bucket, key = path[5:].split("/", 1)
        try:
            resp = s3.get_object(Bucket=bucket, Key=key + "model_latest.pth")
            buf = io.BytesIO(resp["Body"].read())
            return torch.load(buf, map_location="cpu", weights_only=False)
        except ClientError:
            return None
    else:
        ckpt_path = os.path.join(path, "model_latest.pth")
        if os.path.exists(ckpt_path):
            return torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return None
