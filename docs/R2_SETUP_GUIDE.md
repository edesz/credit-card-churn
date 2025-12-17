# Cloudflare R2 Storage Setup Guide

## Overview

The project uses Cloudflare R2 (S3-compatible storage) to store datasets:
- Bucket Name: `cc-churn-splits`
- Training data: `train_data.parquet.gzip`
- Validation data: `validation_data.parquet.gzip`

## Setup Instructions

### 1. Create Environment File

Create a `.env` file in the project root:

```bash
cp docs/env_template.txt .env
```

### 2. Add Credentials

Edit `.env` with your R2 credentials:

```env
ACCOUNT_ID=your_cloudflare_account_id
ACCESS_KEY_ID_USER2=your_access_key_id
SECRET_ACCESS_KEY_USER2=your_secret_access_key
```

Contact your team lead (Elstan) to obtain these credentials.

### 3. Verify Connection

```bash
python scripts/test_r2_connection.py
```

Expected output: All tests should pass.

## Usage in Notebooks

Notebooks automatically detect `.env` and load R2 data:

```python
from dotenv import load_dotenv
import os

env_loaded = load_dotenv()
if env_loaded:
    # Load from R2
    df = pandas_read_parquet_r2('cc-churn-splits', 'train_data.parquet.gzip')
else:
    # Use synthetic data
    df = create_sample_data()
```

## Security

- Keep `.env` file secure
- Never commit `.env` to Git (already in `.gitignore`)
- Use different credentials per team member
- Rotate credentials periodically

## Troubleshooting

### "403 Access Denied"
Verify credentials are correct and you have read access to the bucket.

### ".env file not found"
Ensure the file is in the project root (same directory as `README.md`).

### "No module named 'dotenv'"
Install required packages:
```bash
pip install python-dotenv boto3
```

## Alternative: Demo Data

If you don't have R2 access, notebooks will automatically generate synthetic data that mimics the real dataset structure.
