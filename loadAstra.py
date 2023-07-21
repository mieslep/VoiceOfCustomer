import os
from dotenv import load_dotenv, find_dotenv
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Value
from CONSTANTS import *

load_dotenv(find_dotenv(), override=True)

embed_file = f"{example_asin}.reviews-embeddings-{embed_model}.parquet.gz"
df = pd.read_parquet(embed_file)

# Astra connection
cloud_config = {'secure_connect_bundle': os.environ['ASTRA_SECUREBUNDLE_PATH']}
auth_provider = PlainTextAuthProvider(os.environ['ASTRA_CLIENT_ID'], os.environ['ASTRA_CLIENT_SECRET'])
cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider)
session = cluster.connect()

# Multi-threaded load
processed_counter = Value('i', 0)
error_counter = Value('i', 0)
retry_counter = Value('i', 0)
db_init_counter = Value('i', 0)

def check_and_close_init_bar():
    with db_init_counter.get_lock():
        if db_init_counter.value == num_threads:
            db_init_progress.close()

session.execute(f"""
    CREATE TABLE IF NOT EXISTS {keyspace_name}.{table_name} 
    (asin text
    ,reviewer_id text
    ,unix_review_time timestamp
    ,reviewer_name text
    ,overall_rating int
    ,review_text text
    ,truncated_review_text text
    ,embedding_vector VECTOR<FLOAT, {embed_dimensions}>
    ,PRIMARY KEY((asin, reviewer_id), unix_review_time))
""")

session.execute(f"""
    CREATE CUSTOM INDEX IF NOT EXISTS {table_name}_ann 
    ON {keyspace_name}.{table_name} (embedding_vector) 
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' 
    WITH OPTIONS = {{ 'similarity_function': 'dot_product' }}
""")

session.execute(f"""
    CREATE CUSTOM INDEX IF NOT EXISTS {table_name}_asin
    ON {keyspace_name}.{table_name} (asin) 
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' 
""")

session.execute(f"""
    CREATE CUSTOM INDEX IF NOT EXISTS {table_name}_rating
    ON {keyspace_name}.{table_name} (overall_rating) 
    USING 'org.apache.cassandra.index.sai.StorageAttachedIndex' 
""")

class DB:
    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return super().default(obj)
    
    def __init__(self, cluster: Cluster):
        self.session = cluster.connect()
        self.pinsert = self.session.prepare(f"""
            INSERT INTO {keyspace_name}.{table_name}
            (asin, reviewer_id, unix_review_time, reviewer_name, overall_rating, review_text, truncated_review_text, embedding_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """)
        self.encoder = self.JSONEncoder()
        with db_init_counter.get_lock():
            db_init_counter.value += 1
            db_init_progress.update()
            check_and_close_init_bar()

    def upsert_one(self, row):
        self.session.execute(self.pinsert, [
                row['asin'],
                row['reviewerID'],
                row['unixReviewTime'],
                row['reviewerName'],
                row['overall'],
                row['reviewText'],
                row['truncated'],
                row['embeddings']]
        )

thread_local_storage = threading.local()

def get_db():
    if not hasattr(thread_local_storage, 'db_handle'):
        thread_local_storage.db_handle = DB(cluster)
    return thread_local_storage.db_handle

def upsert_row(indexed_row):
    _, row = indexed_row  # unpack tuple
    db = get_db()
    row = row.to_dict()
    row['embeddings'] = row['embeddings'].tolist()

    # Wrap the database operation and counter increment in a try/except block
    retries = 5
    loaded = False
    tryCount = 0
    while not loaded:
        try:
            db.upsert_one(row)
            with processed_counter.get_lock():  # ensure thread-safety with a lock
                processed_counter.value += 1
            loaded = True
        except Exception as e:
            if tryCount < retries:
                print(f"Error processing row: {e}. Retrying...")
                tryCount += 1
                with retry_counter.get_lock():  # ensure thread-safety with a lock
                    retry_counter.value += 1
                time.sleep(1)
            else:
                with error_counter.get_lock():  # ensure thread-safety with a lock
                    error_counter.value += 1
                print(f"Error processing row: {e}. Fatal.")  
                loaded = True

num_threads = 25

db_init_progress = tqdm(total=num_threads, desc="Thread Initialization")

# Initialize a single progress bar with total number of records
progress_bar = tqdm(total=df.shape[0], desc="Record Loading Progress")

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(upsert_row, indexed_row) for indexed_row in df.iterrows()]
    for future in as_completed(futures):
        future.result()  # this raises any exceptions that occurred in the function
        progress_bar.update()
        
# Close the progress bar when all tasks are complete
progress_bar.close()  

# After all the data is processed
print(f"Total rows processed: {processed_counter.value}")
print(f"Retries: {retry_counter.value}")
print(f"Error rows: {error_counter.value}")
