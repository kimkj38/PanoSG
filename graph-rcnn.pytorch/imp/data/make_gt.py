import json
import os
import numpy as np

object_path = '/tmp/object_json'
wall_path = '/tmp/wall_json'

json_list = os.listdir(object_path)

# object_json과 wall_json 합쳐주기
def merge_json(object_path, wall_path, json_list):
    for filename in json_list:
        object_file = os.path.join(object_path, filename)
        wall_file = os.path.join(wall_path, filename)
        save_path = os.path.join('/tmp', 'merged_json', filename)

        with open(object_file) as o:
            obj = json.load(o)

        with open(wall_file) as w:
            wall = json.load(w)

        merge = dict(wall, **obj)

        with open(save_path, 'w') as out:
            json.dump(merge, out, indent='\t')

#merge_json(object_path, wall_path, json_list)

def count_label(merge_path):
    merged_list = os.listdir(merge_path)
    label_list = []

    for file in merged_list:
        file_path = os.path.join(merge_path, file)
        with open(file_path, 'r') as f:
            merge_json = json.load(f)
        
        for key in merge_json.keys():
            label = merge_json[key]['label']
            label_list.append(label)

    total_boxes = len(label_list) # 전체 박스의 수
    classes = set(label_list) #클래스 종류
    classes_count = len(set(label_list)) # 클래스의 수

    return total_boxes, classes, classes_count

# merge_path = '/tmp/merged_json'
# total_boxes, classes, classes_count = count_label(merge_path)


# 새로운 인덱스로 바꿔주기
def reindex(classes, path, type='wall'):
    
    # key: 이전 index, value: 새로운 index
    reindex_dict = {}
    
    for new, old in enumerate(classes):
        reindex_dict[old] = new

    file_list = os.listdir(path)
    print(reindex_dict)

    for file in file_list:
        file_path = os.path.join(path, file)
        with open(file_path, 'r') as f:
            file_json = json.load(f)

        #key_name = file[5:-5]
        #file_json = file_json[key_name]

        for key in file_json.keys():
            new_index = reindex_dict[file_json[key]['label']]
            file_json[key]['label'] = new_index

        if type == 'wall':     
            save_path = os.path.join('/tmp/wall_json', file)
        elif type == 'object':
            save_path = os.path.join('/tmp/object_json', file)
        elif type == 'merge':
            save_path = os.path.join('/tmp/merged_json', file)

        with open(save_path, 'w') as out:
            json.dump(file_json, out, indent='\t')

# reindex(classes, wall_path)
# reindex(classes, object_path, 'object')
# reindex(classes, merge_path, 'merge')

# wall의 최대 개수->9개
def max_wall_count():
    wall_path = '/tmp/wall_json'
    json_list = os.listdir(object_path)

    max_wall = 0

    for file in json_list:
        file_path = os.path.join(wall_path, file)
        with open(file_path, 'r') as f:
            merge_json = json.load(f)

        key_name = file[5:-5]
        wall_count = len(merge_json[key_name].keys())-2
        if wall_count > max_wall:
            max_wall = wall_count
    print(max_wall)


"""
left: 0, right: 1, include: 2, belong: 3, None: 4

"""

# relation triplet 만들기
def make_triplets():
    relation_triplets = []

    i=0
    for filename in json_list:
        
        object_file = os.path.join(object_path, filename)
        wall_file = os.path.join(wall_path, filename)
        merge_file = os.path.join(merge_path, filename)

        with open(object_file) as o:
            obj = json.load(o)

        with open(wall_file) as w:
            wall = json.load(w)

        with open(merge_file) as m:
            merge = json.load(m)    
        
        # wall과 boundary에 대한 triplet
        for i, (key, value) in enumerate(wall.items()):
            
            # boundary 1
            if i == 0:
                triplet1 = np.array((list(wall.values())[i]['label'], list(wall.values())[i+2]['label'], 3), dtype='int64')
                triplet2 = np.array((list(wall.values())[i+2]['label'], list(wall.values())[i]['label'], 2), dtype='int64')
                relation_triplets.append(triplet1)
                relation_triplets.append(triplet2)

            # boundary 2
            elif i == 1:
                triplet1 = np.array((list(wall.values())[i]['label'], list(wall.values())[-1]['label'], 3), dtype='int64')
                triplet2 = np.array((list(wall.values())[-1]['label'], list(wall.values())[i]['label'], 2), dtype='int64')
                relation_triplets.append(triplet1)
                relation_triplets.append(triplet2)

            # walls
            elif i < (len(wall)-1):
                triplet1 = np.array((list(wall.values())[i]['label'], list(wall.values())[i+1]['label'], 0), dtype='int64')
                triplet2 = np.array((list(wall.values())[i+1]['label'], list(wall.values())[i]['label'], 1), dtype='int64')
                relation_triplets.append(triplet1)
                relation_triplets.append(triplet2)
        
        

        wall_x = [] # wall-wall boundary의 x좌표
        obj_cx = {} # key: object id, value: center x좌표
        for i, (key, value) in enumerate(merge.items()):        
            if "wall" in key:
                wall_x.append(merge[key]['bbox'][0])

            if "obj" in key:
                bbox = value['bbox']
                cx = bbox[0] + ((bbox[2] - bbox[0])/2)
                obj_cx[key] = cx
        wall_x.append(np.inf)


        # obj_cx를 value에 대한 오름차순 정렬
        obj_cx = sorted(obj_cx.items(), key=lambda x:x[1])
        obj_cx_sort = {}
        for k, v in obj_cx:
            obj_cx_sort[k] = v


        # 정렬에 따라 object 간의 triplet 만들기
        for i, (key, value) in enumerate(obj_cx_sort.items()):
            if i < (len(obj_cx_sort)-1):
                triplet1 = np.array((merge[list(obj_cx_sort.keys())[i]]['label'], merge[list(obj_cx_sort.keys())[i+1]]['label'], 0), dtype='int64')
                triplet2 = np.array((merge[list(obj_cx_sort.keys())[i+1]]['label'], merge[list(obj_cx_sort.keys())[i]]['label'], 1), dtype='int64')
                relation_triplets.append(triplet1)
                relation_triplets.append(triplet2)

        # object와 boundary 간의 triplet 만들기
        if obj_cx_sort:
            first_obj_label = int(merge[list(obj_cx_sort.keys())[0]]['label'])
            last_obj_label = int(merge[list(obj_cx_sort.keys())[-1]]['label'])
            first_obj_cx = list(obj_cx_sort.values())[0]
            last_obj_cx = list(obj_cx_sort.values())[-1]

            boundary0_label = int(merge[list(merge.keys())[0]]['label'])
            boundary1_label = int(merge[list(merge.keys())[1]]['label'])
            boundary0_bbox = merge[list(merge.keys())[0]]['bbox']
            boundary1_bbox = merge[list(merge.keys())[1]]['bbox']

            # object가 boundary 영역 안에 있으면 include, belong 아니면 left, right
            if boundary0_bbox[0] < first_obj_cx < boundary0_bbox[2]:
                triplet1 = np.array((first_obj_label, boundary0_label, 3), dtype='int64')
                triplet2 = np.array((boundary0_label, first_obj_label, 2), dtype='int64')
            else:
                triplet1 = np.array((first_obj_label, boundary0_label, 1), dtype='int64')
                triplet2 = np.array((boundary0_label, first_obj_label, 0), dtype='int64')

            if boundary1_bbox[0] < last_obj_cx < boundary1_bbox[2]:
                triplet3 = np.array((last_obj_label, boundary1_label, 3), dtype='int64')
                triplet4 = np.array((boundary1_label, last_obj_label, 2), dtype='int64')
            else:
                triplet3 = np.array((last_obj_label, boundary1_label, 1), dtype='int64')
                triplet4 = np.array((boundary1_label, last_obj_label, 0), dtype='int64')
            
            relation_triplets.append(triplet1)
            relation_triplets.append(triplet2)
            relation_triplets.append(triplet3)
            relation_triplets.append(triplet4)
        

        # object와 wall 간의 triplet 만들기
        for i, (key, value) in enumerate(merge.items()):
            if "obj" in key:
                bbox = value['bbox']
                cx = bbox[0] + ((bbox[2] - bbox[0])/2)

                for j, x in enumerate(wall_x):
                    if cx < x:
                        if i == 0:
                            wall_idx = "wall_{}".format(j)
                        else:
                            wall_idx = "wall_{}".format(j-1)
                        triplet1 = [merge[wall_idx]['label'], merge[key]['label'], 2]
                        triplet2 = [merge[key]['label'], merge[wall_idx]['label'], 3]
                        relation_triplets.append(triplet1)
                        relation_triplets.append(triplet2)
                        break

    return relation_triplets

# relation_triplets = make_triplets()
# print(len(relation_triplets))

def get_img_ids(data_dir, split):
    folder = os.path.join(data_dir, 'mask_data', split)
    return os.listdir(folder)




        




