import json
from importlib import metadata
from urllib.parse import unquote_plus, urlparse
import numpy as np
import os
import requests
from tqdm import tqdm

def resolve_data_path(data_path):
    if data_path != None:
        return data_path

    dist_info = json.loads(metadata.distribution('noisebase').read_text('direct_url.json'))
    editable = dist_info.get('dir_info', {}).get('editable', False)

    if editable:
        nb_folder = unquote_plus(urlparse(dist_info['url']).path)
        return os.path.join(nb_folder, 'data')
    else:
        return os.path.join(os.getcwd(), 'data')

def consumer_generator(queue, producer_process):
    try:
        while True:
            if producer_process.exitcode is not None and queue.empty():
                # Handle producer crash
                break
            item = queue.get()
            if item is None: # Producer successfully finished
                break
            yield item
    finally:
        # cleanup
        if producer_process.is_alive():
            producer_process.terminate()
        producer_process.join()

# Adapted from https://github.com/TheRealMJP/BakingLab/blob/master/BakingLab/ACES.hlsl
def ACES(x, gamma_correct = True, gamma = 2.2):

    ACESInputMat = np.array([
        [0.59719, 0.35458, 0.04823],
        [0.07600, 0.90834, 0.01566],
        [0.02840, 0.13383, 0.83777]
    ])

    ACESOutputMat = np.array([
        [1.60475, -0.53108, -0.07367],
        [-0.10208,  1.10813, -0.00605],
        [-0.00327, -0.07276,  1.07602]
    ])

    x = np.einsum('ji, hwi -> hwj', ACESInputMat, x)
    a = x * (x + 0.0245786) -  0.000090537
    b = x * (0.983729 * x + 0.4329510) + 0.238081
    x = a / b
    x = np.einsum('ji, hwi -> hwj', ACESOutputMat, x)

    if gamma_correct:
        return np.power(np.clip(x, 0.0, 1.0), 1.0/gamma)
    else:
        return x

def downloader(local_files, remote_files):
    '''
    Downloading is deliberately limited to a single connection. 
    Our data formats don’t require small files, so this does not 
    waste bandwidth. If you get faster downloads by modifying 
    the script, that will come at others’ expense. Please be 
    patient and mindful of others.
    '''
    for local, remote in tqdm(zip(local_files, remote_files), total=len(local_files), unit='files', desc='Downloading'):
        if os.path.exists(local):
            continue # Only keep complete files with the original name

        os.makedirs(os.path.dirname(local), exist_ok=True)
        
        temp = local+'.part'
        if os.path.exists(temp):
            os.remove(temp) # Partial resume not supported
        
        with open(temp, 'wb') as f, requests.get(remote, stream=True) as r:
            r.raise_for_status()
            MBs = int(r.headers.get('content-length')) // 1024 // 1024

            for chunk in tqdm(r.iter_content(chunk_size=1024 * 1024), leave=False, total=MBs, unit='MB', desc=os.path.basename(local)):
                f.write(chunk)
            
            os.rename(temp, local)