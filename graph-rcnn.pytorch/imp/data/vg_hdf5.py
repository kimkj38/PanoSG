import os
from collections import defaultdict
import numpy as np
import copy
import pickle
import scipy.sparse
from PIL import Image
import h5py, json
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from lib.scene_parser.rcnn.structures.bounding_box import BoxList
from lib.utils.box import bbox_overlaps
from lib.data.make_gt import make_triplets, get_img_ids, count_label

class vg_hdf5(Dataset):
    def __init__(self, cfg, split="train", transforms=None, num_im=-1):

        assert split == "train" or split == "test", "split must be one of [train, val, test]"
        assert num_im >= -1, "the number of samples must be >= 0"

        self.transforms = transforms

        self.split = split
        #self.data_dir = cfg.DATASET.PATH
        self.data_dir = "/tmp"

        assert os.path.exists(self.data_dir), \
            "cannot find folder {}, please download the visual genome data into this folder".format(self.data_dir)

        self.info = json.load(open(os.path.join(self.data_dir, "class_index.json"), 'r'))

        # filename이 담긴 리스트(mask_pano_00000.png)
        self.image_index = get_img_ids(self.data_dir, self.split)

        # node lable-index
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        cfg.ind_to_class = self.ind_to_classes

        # relation label-index
        self.predicate_to_ind = self.info['predicate_to_idx']
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])
        cfg.ind_to_predicate = self.ind_to_predicates

        self.json_category_id_to_contiguous_id = self.class_to_ind

        self.contiguous_category_id_to_json_id = {
            v: k for k, 
            v in self.json_category_id_to_contiguous_id.items()
        }

        self.total_boxes, self.classes, self.classes_count = count_label(os.path.join(self.data_dir, "merged_json"))

    def get_filelist(self):
        path_list = []

        if self.split == "train":
            path = os.path.join(self.data_dir, "mask_data/train")
        else:
            path = os.path.join(self.data_dir, "mask_data/test")
        file_list = os.listdir(path)
        for filename in file_list:
            file_path = os.path.join(path, filename)
            path_list.append(file_path)

        return path_list

    # 인덱스 통해 이미지 경로 받기
    def get_img_path(self, idx):
        filename = "mask_pano_{:05d}.png".format(idx)
        if self.split == "train":
            file_path = os.path.join(self.data_dir, "mask_data/train", filename)
        else:
            file_path = os.path.join(self.data_dir, "mask_data/test", filename)

        return file_path

    # 이미지 인덱스 통해서 해당 이미지의 boxes, labels 받기    
    def get_boxes_labels(self, idx):
        boxes = []
        labels = []
        filename = "mask_pano_{:05d}.json".format(idx)
        file_path = os.path.join(self.data_dir, "merged_json", filename)
        
        with open(file_path, 'r') as f:
            merge_json = json.load(f)
        
        for node in merge_json.keys():
            box = merge_json[node]['bbox']
            label = merge_json[node]['label']
            boxes.append(box)
            labels.append(label)
        
        return boxes, labels

    def __len__(self):
        return len(self.image_index)

    def __getitem__(self, index):
        """
        get dataset item
        """
        global image, target
        # 해당 인덱스의 이미지
        file_num = int(self.get_filelist()[index][-9:-4])
        image = Image.open(self.get_filelist()[index]); width, height = image.size

        # target에 들어갈 정보들
        obj_boxes, obj_labels = self.get_boxes_labels(file_num)
        obj_relation_triplets = make_triplets(self.data_dir, file_num) # (subj, obj, relation)
        obj_relation_triplets = np.array(obj_relation_triplets)
        obj_labels = np.array(obj_labels)

        total_boxes, classes, classes_count = count_label(os.path.join(self.data_dir, "merged_json"))

        # class 종류의 size를 가진 정사각 array
        obj_relations = np.zeros((classes_count, classes_count))

        # triplet의 개수만큼 반복문
        # 행이 subj, 열이 obj일 때 값은 relation
        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        # Boxlist에 담아주기
        target_raw = BoxList(obj_boxes, (width, height), mode="xyxy")
        image, target = self.transforms(image, target_raw)
        target.add_field("labels", torch.from_numpy(obj_labels))
        target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
        target = target.clip_to_image(remove_empty=False)

        return image, target, index

    def get_groundtruth(self, index):

        file_num = int(self.get_filelist()[index][-9:-4])
        
        # target에 들어갈 정보들
        obj_boxes, obj_labels = self.get_boxes_labels(file_num)
        obj_relation_triplets = make_triplets(self.data_dir) # (subj, obj, relation)
        obj_relation_triplets = np.array(obj_relation_triplets, dtype='int64')
        obj_labels = np.array(obj_labels)

        total_boxes, classes, classes_count = count_label(os.path.join(self.data_dir, "merged_json"))

        # class 종류의 size를 가진 정사각 array
        obj_relations = np.zeros(classes_count, classes_count)

        # triplet의 개수만큼 반복문
        # 행이 subj, 열이 obj일 때 값은 relation
        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        # Boxlist에 담아주기
        target_raw = BoxList(obj_boxes, (width, height), mode="xyxy")
        img, target = self.transforms(img, target_raw)
        target.add_field("labels", torch.from_numpy(obj_labels))
        target.add_field("pred_labels", torch.from_numpy(obj_relations))
        target.add_field("relation_labels", torch.from_numpy(obj_relation_triplets))
        target = target.clip_to_image(remove_empty=False)

        return target

    def get_img_info(self, img_id):
        w, h = 2048, 1024
        return {"height": h, "width": w}

    def map_class_id_to_class_name(self, class_id):
        return self.ind_to_classes[class_id]

    @property
    def coco(self):
        """
        :return: a Coco-like object that we can use to evaluate detection!
        """
        anns = []
        for i, (cls_array, box_array) in enumerate(zip(self.gt_classes, self.gt_boxes)):
            for cls, box in zip(cls_array.tolist(), box_array.tolist()):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': i,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'ayy lmao'},
            'images': [{'id': i} for i in range(self.__len__())],
            'categories': [{'supercategory': 'person',
                               'id': i, 'name': name} for i, name in enumerate(self.ind_to_classes) if name != '__background__'],
            'annotations': anns,
        }
        fauxcoco.createIndex()
        return fauxcoco