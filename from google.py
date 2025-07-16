from google.cloud import storage

client = storage.Client()
bucket = client.bucket("your-bucket-name")
blob = bucket.blob("file.txt")
blob.upload_from_string("Hello, World!")