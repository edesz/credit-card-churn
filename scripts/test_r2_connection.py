"""
Test script to verify R2 connection and data access.
Run this script after setting up your .env file to ensure everything is configured correctly.
"""

import os
import sys
from pathlib import Path
from io import BytesIO
import pandas as pd
from dotenv import load_dotenv

def test_r2_connection():
    """Test R2 connection and data loading"""
    
    print("=" * 60)
    print("R2 CONNECTION TEST")
    print("=" * 60)
    
    # Load environment variables
    print("\n1. Loading environment variables from .env...")
    env_loaded = load_dotenv()
    
    if not env_loaded:
        print("   [ERROR] .env file not found!")
        print("   Please create a .env file with your R2 credentials.")
        print("   See R2_SETUP_GUIDE.md for instructions.")
        return False
    
    print("   [OK] .env file loaded successfully")
    
    # Check credentials
    print("\n2. Checking credentials...")
    account_id = os.getenv('ACCOUNT_ID')
    access_key_id = os.getenv('ACCESS_KEY_ID_USER2')
    secret_access_key = os.getenv('SECRET_ACCESS_KEY_USER2')
    
    if not account_id:
        print("   [ERROR] ACCOUNT_ID not found in .env")
        return False
    if not access_key_id:
        print("   [ERROR] ACCESS_KEY_ID_USER2 not found in .env")
        return False
    if not secret_access_key:
        print("   [ERROR] SECRET_ACCESS_KEY_USER2 not found in .env")
        return False
    
    print(f"   [OK] ACCOUNT_ID: {account_id[:8]}...{account_id[-8:]}")
    print(f"   [OK] ACCESS_KEY_ID_USER2: {access_key_id[:8]}...{access_key_id[-8:]}")
    print(f"   [OK] SECRET_ACCESS_KEY_USER2: {'*' * 20}")
    
    # Test boto3 import
    print("\n3. Importing boto3...")
    try:
        import boto3
        print("   [OK] boto3 imported successfully")
    except ImportError:
        print("   [ERROR] boto3 not found. Install it with: pip install boto3")
        return False
    
    # Create S3 client
    print("\n4. Creating S3 client for R2...")
    try:
        s3_client = boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name='auto'
        )
        print("   [OK] S3 client created successfully")
    except Exception as e:
        print(f"   [ERROR] Failed to create S3 client: {e}")
        return False
    
    # Test bucket access
    print("\n5. Testing bucket access...")
    bucket_name = 'cc-churn-splits'
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=10)
        key_count = response.get('KeyCount', 0)
        print(f"   [OK] Successfully accessed bucket: {bucket_name}")
        print(f"   [OK] Found {key_count} objects in bucket")
        
        if 'Contents' in response:
            print("\n   Available files:")
            for obj in response['Contents']:
                size_mb = obj['Size'] / (1024 * 1024)
                print(f"      - {obj['Key']} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"   [ERROR] Failed to access bucket: {e}")
        return False
    
    # Test data loading
    print("\n6. Testing data loading...")
    r2_key_train = 'train_data.parquet.gzip'
    try:
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=r2_key_train)
        df = pd.read_parquet(
            BytesIO(s3_object['Body'].read()), 
            dtype_backend='pyarrow'
        )
        print(f"   [OK] Successfully loaded {r2_key_train}")
        print(f"   [OK] Data shape: {df.shape}")
        print(f"   [OK] Columns: {', '.join(df.columns[:5])}...")
        
        if 'is_churned' in df.columns:
            churn_rate = df['is_churned'].mean()
            print(f"   [OK] Churn rate: {churn_rate:.1%}")
    except Exception as e:
        print(f"   [ERROR] Failed to load data: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL TESTS PASSED!")
    print("=" * 60)
    print("\nYou're ready to use R2 data in your notebooks!")
    print("Run: jupyter lab notebooks/04_advanced_modeling.ipynb")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_r2_connection()
    sys.exit(0 if success else 1)

