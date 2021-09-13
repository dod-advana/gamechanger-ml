## SOURCE: based on source code from https://gist.github.com/feelinc/d1f541af4f31d09a2ec3
#!/usr/bin/python

import os
import sys
import boto3

# get an access token, local (from) directory, and S3 (to) directory
# from the command-line
local_directory, bucket, destination = sys.argv[1:4]

client = boto3.client('s3')

# enumerate local files recursively
for root, dirs, files in os.walk(local_directory):

  for filename in files:

    # construct the full local path
    local_path = os.path.join(root, filename)

    # construct the full Dropbox path
    relative_path = os.path.relpath(local_path, local_directory)
    s3_path = os.path.join(destination, relative_path)

    print('Searching "%s" in "%s"' % (s3_path, bucket))
    try:
        client.head_object(Bucket=bucket, Key=s3_path)
        print("Path found on S3! Skipping %s..." % s3_path)

    except:
        print("Uploading %s..." % s3_path)
        client.upload_file(local_path, bucket, s3_path)