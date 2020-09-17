import os
import argparse

ROOT_URL = 'http://abstraction.csail.mit.edu/data/metadata'

FILENAME_TMPLS = {
    'graph': '{}_graph.json',
    'category_embeddings': '{}_category_embeddings.pth',
    'listfile': '{}_{}_listfile.txt',
    'abstraction_sets': '{}_{}_abstraction_sets.json',
    'embeddings': '{}_{}_embeddings.pth',
    'features': 'resnet3d50_{}_{}_features.pth',
    'pretrained_model': 'AbstractionEmbedding_{}_SAM_1-2-3-4_best.pth.tar',
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Set Abstraction Metadata Download Utility."
    )

    d_defaults = ['moments', 'kinetics']
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        choices=d_defaults,
        default=d_defaults,
        help='list of datasets for which to download metadata. '
        '(default: %(default)s)',
    )

    t_defaults = [
        'abstraction_sets',
        'embeddings',
        'category_embeddings',
        'features',
        'graph',
        'listfile',
        'pretrained_model'
    ]
    parser.add_argument(
        '--datatypes',
        type=str,
        nargs='+',
        choices=t_defaults,
        default=t_defaults,
        help='list of metadata types to download. (default: %(default)s)',
    )

    s_defaults = ['train', 'val', 'test']
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        choices=s_defaults,
        default=s_defaults,
        help='list of splits for which to download metadata (default: %(default)s)',
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='metadata',
        help='directory to save downloaded data (default: %(default)s)',
    )
    parser.add_argument(
        '--overwrite', action='store_true', help='overwrite current data if it exists',
    )
    args = parser.parse_args()
    return args


def download_url(url, outdir='.', overwrite=True):
    outfile = os.path.join(outdir, os.path.basename(url))
    cmd = 'wget {} -O {}'.format(url, outfile)
    if os.path.exists(outfile) and not overwrite:
        print('Skipping download since {} exists already...'.format(outfile))
        return
    os.system(cmd)


def download_data(
    datasets, datatypes, splits, outdir='metadata', overwrite=False,
):
    os.makedirs(outdir, exist_ok=True)
    for dset in datasets:
        for dtype in datatypes:
            file_tmpl = FILENAME_TMPLS[dtype]
            if dtype in ['graph', 'category_embeddings', 'pretrained_model']:
                filename = file_tmpl.format(dset)
                url = os.path.join(ROOT_URL, filename)
                download_url(url, outdir, overwrite=overwrite)
            else:
                for split in splits:
                    if split == 'test' and dtype == 'abstraction_sets':
                        split = 'human'
                    filename = file_tmpl.format(dset, split)
                    url = os.path.join(ROOT_URL, filename)
                    download_url(url, outdir, overwrite=overwrite)


if __name__ == '__main__':
    args = parse_args()
    download_data(
        args.datasets,
        args.datatypes,
        args.splits,
        outdir=args.data_dir,
        overwrite=args.overwrite,
    )
