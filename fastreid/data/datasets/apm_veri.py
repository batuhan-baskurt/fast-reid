# encoding: utf-8

import glob
import os.path as osp
import re

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class APM_VeRi(ImageDataset):
    """APM_VeRi.

    Dataset statistics:
        - identities: 2728.
        - images: 37778 (train) + 1678 (query) + 11579 (gallery).
    """
    dataset_dir = "APM_VeRi"
    dataset_name = "APM_VeRi"

    def __init__(self, root='datasets', **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'bbox_train')
        self.query_dir = osp.join(self.dataset_dir, 'bbox_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bbox_test')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)

        super(APM_VeRi, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c(\d\d\d)')

        data = []
        for img_path in img_paths:
            #pid, camid = map(int, pattern.search(img_path).groups())
            elems = img_path.split('/')
            if not elems:
                continue
            file_name = elems[-1]
            pid = int(file_name[0:4])
            # TODO: does not work for two digit camid
            camid = int(file_name[5])
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 2729
            assert 1 <= camid <= 20
            camid -= 1  # index starts from 0
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
                camid = self.dataset_name + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data
