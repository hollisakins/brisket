
import boto3
import sys, os
from botocore import UNSIGNED
from botocore.client import Config
from brisket import config

class GridManager:
    def __init__(self, bucket='brisket-data'):
        self.bucket = bucket
        #create the s3 client and assign credentials (UNSIGEND for public bucket)
        self.client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def list_objects(self):
        objs = self.client.list_objects(Bucket=self.bucket)['Contents']
        return objs

    def download_file(self, key, local_path):
        meta = self.client.head_object(Bucket=self.bucket, Key=key)
        size = int(meta.get('ContentLength', 0))
        downloaded = 0
        
        if size > 1e6:
            print(f'Downloading {size/1e6:.1f} MB')
        elif size > 1e9:
            print(f'Downloading {size/1e9:.1f} GB')
        else:
            print(f'Downloading {size/1e3:.1f} kB')

        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            done = int(50 * downloaded / size)
            sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50-done)) )
            sys.stdout.flush()

        self.client.download_file(self.bucket, key, local_path, Callback=progress)

    def check_grid(self, grid_file_name):
        if os.path.exists(os.path.join(config.grid_dir, grid_file_name)):
            #print('We have the grid locally, nothing to do')
            pass
        else:
            # obj = self.client.get_object(Bucket=self.bucket, Key=grid_file_name)
            # obj = self.client.download_file(Bucket=self.bucket, Key=grid_file_name)
            print(f"Grid file {grid_file_name} not found locally at {config.grid_dir}")
            whether_continue = input(f"Do you want to fetch the latest version from s3://{self.bucket}? (y/n): ")
            if whether_continue.lower() == 'y':
                print(f'Downloading grid {grid_file_name}')
                self.download_file(grid_file_name, os.path.join(config.grid_dir, grid_file_name))
        



gm = GridManager()
gm.check_grid('d_igm_grid_inoue14.fits')

# print(dir(s3))

#     for key in list_files:
#         if key['Key'].endswith('.zip'):
#             print(f'downloading... {key["Key"]}') #print file name
#             client.download_file(
#                                     Bucket=bucket, #assign bucket name
#                                     Key=key['Key'], #key is the file name
#                                     Filename=os.path.join('./data', 
#                                         key['Key']) #storage file path
#                                 )
#         else:
#             pass #if it's not a zip file do nothing

# get_s3_public_data()

