from noisebase import Noisebase
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export reference images from test datasets for metrics computation.")
    parser.add_argument('dataset', type=str, help='Input test dataset')
    parser.add_argument('--data_path', type=str, default=None, help='Custom data path')
    args = parser.parse_args()

    test_set = Noisebase(args.dataset, {
        'data_path': args.data_path
    })
    test_set.save_reference()