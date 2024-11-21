
import boto3
import sys, os
from botocore import UNSIGNED
from botocore.client import Config
from brisket import config

from rich import print
from rich.prompt import Confirm

class GridManager:
    def __init__(self, bucket='brisket-data'):
        self.bucket = bucket
        #create the s3 client and assign credentials (UNSIGEND for public bucket)
        self.client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    def list_objects(self):
        objs = self.client.list_objects(Bucket=self.bucket)['Contents']
        return objs

    def get_size(self, key):
        meta = self.client.head_object(Bucket=self.bucket, Key=key)
        return int(meta.get('ContentLength', 0))

    def format_size(self, size):
        if size > 1e6:
            return f'{size/1e6:.1f}MB'
        elif size > 1e9:
            return f'{size/1e9:.1f}GB'
        else:
            return f'{size/1e3:.1f}kB'


    def download_file(self, key, local_path):
        size = self.get_size(key)
        downloaded = 0
        pbar_size = config.cols - len(key) - 17
        
        def progress(chunk):
            nonlocal downloaded
            downloaded += chunk
            done = int(pbar_size * downloaded / size)
            print(f"[blue]{key}[/blue]: [red]{self.format_size(downloaded):<7}[/red] |{'#'*done}{' '*(pbar_size-done)}| "+f"[red]{int(done/pbar_size*100)}%[/red]".ljust(4), end='\r', flush=True)

        self.client.download_file(self.bucket, key, local_path, Callback=progress)
        print('\n')

    def check_grid(self, grid_file_name):
        if os.path.exists(os.path.join(config.grid_dir, grid_file_name)):
            #print('We have the grid locally, nothing to do')
            pass
        else:
            # obj = self.client.get_object(Bucket=self.bucket, Key=grid_file_name)
            # obj = self.client.download_file(Bucket=self.bucket, Key=grid_file_name)
            print(f"Grid file [blue]{grid_file_name}[/blue] not found locally at [blue]{config.grid_dir}[/blue]")
            size = self.get_size(grid_file_name)
            whether_continue = Confirm.ask(f"Do you want to fetch the latest version [[red]{self.format_size(size)}[/red]] from [blue]s3://{self.bucket}[/blue]?", default=True)
            assert whether_continue
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

