import os
from argparse import ArgumentParser
import common

parser = ArgumentParser(description='integrate embed.py and evaluate.py')

# Required

parser.add_argument(
    '--experiment_root', required=True,
    help='Location used to store checkpoints and dumped data.')

parser.add_argument(
    '--query_dataset', required=True,
    help='Path to the query dataset csv file.')

parser.add_argument(
    '--gallery_dataset', required=True,
    help='Path to the gallery dataset csv file.')

parser.add_argument(
    '--query_image_root', type=common.readable_directory, required=True,
    help='Path that will be pre-pended to the filenames in the query_dataset csv.')

parser.add_argument(
    '--gallery_image_root', type=common.readable_directory, required=True,
    help='Path that will be pre-pended to the filenames in the gallery_dataset csv.')

parser.add_argument(
    '--query_filename', default=None,
    help='Name of the HDF5 file in which to store the embeddings of query images, relative to'
         ' the `experiment_root` location. If omitted, appends `_embeddings.h5` to the dataset name.')

parser.add_argument(
    '--gallery_filename', default=None,
    help='Name of the HDF5 file in which to store the embeddings of gallery images, relative to'
         ' the `experiment_root` location. If omitted, appends `_embeddings.h5` to the dataset name.')

def main():
    args = parser.parse_args()
    os.system("python embed.py --quiet --experiment_root " + args.experiment_root + " --dataset " + args.query_dataset + " --filename " + args.query_filename + " --image_root " + args.query_image_root)
    os.system("python embed.py --quiet --experiment_root " + args.experiment_root + " --dataset " + args.gallery_dataset + " --filename " + args.gallery_filename + " --image_root " + args.gallery_image_root)
    os.system("python evaluate.py --excluder diagonal --query_dataset " + args.query_dataset + " --query_embeddings " + args.query_filename + " --gallery_dataset " +  args.gallery_dataset + " --gallery_embeddings " +  args.gallery_filename + " --metric euclidean")

if __name__ == '__main__':
    main()