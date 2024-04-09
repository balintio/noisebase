from noisebase import Noisebase
import argparse

def main():
    parser = argparse.ArgumentParser(description="Downloads a dataset.")
    parser.add_argument('dataset', type=str, help='The dataset to download')
    parser.add_argument('--data_path', type=str, default=None, help='Custom data path')
    args = parser.parse_args()

    if args.data_path != None:
        print(f'Using a custom data path: {args.data_path}')
        print('When using the dataset, you\'ll need to provide this as the data_path loader parameter.')
        print('When using other scripts, you\'ll need to provide this as the --data_path argument, just like here.')
            
    dataset = Noisebase(args.dataset, {
        'data_path': args.data_path
    })
    dataset.download()