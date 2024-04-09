from noisebase import Noisebase
import argparse
import os
import json

def digits(x, n):
    return f'{{:.{n - f"{x:f}".find(".")}f}}'.format(x)

def main():
    parser = argparse.ArgumentParser(description="Compute metrics on a test dataset.")
    parser.add_argument('dataset', type=str, help='Input test dataset')
    parser.add_argument('outputs', type=str, help='Output directories of methods (comma separated)')
    parser.add_argument('metrics', type=str, help='Metrics (comma separated)')
    parser.add_argument('--digits', type=int, default=6, help='Number of digits to round results to')
    parser.add_argument('--sep', type=str, default=",", help='Delimiter for output formatting')
    parser.add_argument('--data_path', type=str, default=None, help='Custom data path')
    args = parser.parse_args()

    output_dirs = args.outputs.split(',')
    metrics = args.metrics.split(',')
    sep = args.sep

    test_set = Noisebase(args.dataset, {
        'data_path': args.data_path
    })
    metadata = test_set.test_metadata()
    
    results = {}

    for output_dir in output_dirs:

        to_average = {key: 0.0 for key in metrics}
        N = 0

        if not os.path.isdir(output_dir):
            print(f"The output directory provided does not exist or is not a directory: {output_dir}")
            exit(1)
        
        for sequence in metadata['sequences']:
            with open(os.path.join(output_dir, metadata["metric_format"].format(sequence_name=sequence["name"])), 'r') as f:
                seq_metrics = json.load(f)
            for key in metrics:
                to_average[key] += seq_metrics[key]
            N += 1
        
        for key in metrics:
            to_average[key] /= N
        
        results[output_dir] = to_average
        

    print('------ results ------')
    print(f'name{sep}{sep.join(metrics)}')
    for output_dir in output_dirs:
        print(f'{output_dir}{sep}{sep.join(map(lambda x: digits(x, args.digits),results[output_dir].values()))}')