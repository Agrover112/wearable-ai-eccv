#!/usr/bin/env python3
"""Refresh the scoped ECR Docker token without requiring the AWS CLI."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import hmac
import json
import pathlib
import re
import urllib.request


def credential(text: str, name: str) -> str:
    match = re.search(
        rf"(?m)^\s*(?:export\s+)?{re.escape(name)}\s*=\s*['\"]?([^\s'\"#]+)",
        text,
    )
    if match is None:
        raise RuntimeError(f"missing {name} in credentials file")
    return match.group(1)


def sign(key: bytes, message: str) -> bytes:
    return hmac.new(key, message.encode(), hashlib.sha256).digest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--credentials-file", type=pathlib.Path, required=True)
    parser.add_argument("--docker-config", type=pathlib.Path, required=True)
    parser.add_argument("--region", default="us-east-2")
    args = parser.parse_args()

    text = args.credentials_file.read_text()
    access_key = credential(text, "AWS_ACCESS_KEY_ID")
    secret_key = credential(text, "AWS_SECRET_ACCESS_KEY")
    session_match = re.search(
        r"(?m)^\s*(?:export\s+)?AWS_SESSION_TOKEN\s*=\s*['\"]?([^\s'\"#]+)",
        text,
    )
    session_token = session_match.group(1) if session_match else None

    service = "ecr"
    host = f"api.ecr.{args.region}.amazonaws.com"
    endpoint = f"https://{host}/"
    target = "AmazonEC2ContainerRegistry_V20150921.GetAuthorizationToken"
    payload = b"{}"
    now = dt.datetime.now(dt.timezone.utc)
    amz_date = now.strftime("%Y%m%dT%H%M%SZ")
    date_stamp = now.strftime("%Y%m%d")

    headers = {
        "content-type": "application/x-amz-json-1.1",
        "host": host,
        "x-amz-date": amz_date,
        "x-amz-target": target,
    }
    if session_token:
        headers["x-amz-security-token"] = session_token
    signed_headers = ";".join(sorted(headers))
    canonical_headers = "".join(f"{key}:{headers[key]}\n" for key in sorted(headers))
    canonical_request = "\n".join(
        [
            "POST",
            "/",
            "",
            canonical_headers,
            signed_headers,
            hashlib.sha256(payload).hexdigest(),
        ]
    )
    scope = f"{date_stamp}/{args.region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        [
            "AWS4-HMAC-SHA256",
            amz_date,
            scope,
            hashlib.sha256(canonical_request.encode()).hexdigest(),
        ]
    )
    date_key = sign(("AWS4" + secret_key).encode(), date_stamp)
    region_key = sign(date_key, args.region)
    service_key = sign(region_key, service)
    signing_key = sign(service_key, "aws4_request")
    signature = hmac.new(signing_key, string_to_sign.encode(), hashlib.sha256).hexdigest()
    authorization = (
        f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
        f"SignedHeaders={signed_headers}, Signature={signature}"
    )

    request_headers = {key.title(): value for key, value in headers.items()}
    request_headers["Authorization"] = authorization
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)

    entries = body.get("authorizationData", [])
    if len(entries) != 1:
        raise RuntimeError(f"expected one ECR authorization entry, found {len(entries)}")
    entry = entries[0]
    registry = str(entry["proxyEndpoint"]).removeprefix("https://")
    authorization_token = str(entry["authorizationToken"])

    args.docker_config.mkdir(parents=True, exist_ok=True)
    config_path = args.docker_config / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config.setdefault("auths", {})[registry] = {"auth": authorization_token}
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    config_path.chmod(0o600)

    expiry = entry.get("expiresAt", "unknown")
    print(f"ECR authentication refreshed for {registry}")
    print(f"expires_at={expiry}")
    print(f"docker_config={config_path}")


if __name__ == "__main__":
    main()
