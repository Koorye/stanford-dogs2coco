import json
import os
import xmltodict
from scipy.io import loadmat
from tqdm import tqdm


root = 'D:/Datasets/stanford-dogs'

class IDGenerator(object):
    def __init__(self, start_id=1):
        self.start_id = start_id
        self.id_ = start_id
    
    def get(self):
        id_ = self.id_
        self.id_ += 1
        return id_
    
    def reset(self):
        self.id_ = self.start_id

def load_mat(mat_path):
    res = loadmat(mat_path)
    img_paths, labels = res['file_list'], res['labels']
    img_paths = list(map(lambda x: str(x[0][0]), img_paths))
    labels = list(labels.flatten())

    cats, ids = [], []
    for i in range(len(img_paths)):
        name = '-'.join(img_paths[i].split('/')[0].split('-')[1:])
        id_ = int(labels[i] - 1)
        if id_ not in ids:
            ids.append(id_)
            cats.append(dict(
                id=id_,
                name=name,
            ))
    
    return cats, img_paths

def load_xml(xml_path, cats, img_id_gen, ann_id_gen):
    def cat2id(cats, name):
        for cat in cats:
            if cat['name'] == name:
                return cat['id']
        print(name)
        raise Exception('ID NOT FOUND!')
    
    with open(xml_path) as f:
        xmlstr = f.read()
    res = xmltodict.parse(xmlstr)['annotation']
    parent_path = xml_path.split('/')[-2]
    img_path = os.path.join(parent_path, res['filename']).replace('\\', '/') + '.jpg'
    img_width = int(res['size']['width'])
    img_height = int(res['size']['height'])
    img_id = img_id_gen.get()
    objects = res['object']

    anns = []
    if not isinstance(objects, list):
        objects = [objects]
    
    for obj in objects:
        cat_name = obj['name']
        cat_id = cat2id(cats, cat_name)
        bbox = obj['bndbox']
        bbox = [float(bbox['xmin']), 
                float(bbox['ymin']), 
                float(bbox['xmax']), 
                float(bbox['ymax'])]
        bbox[2] = bbox[2] - bbox[0]
        bbox[3] = bbox[3] - bbox[1]
        area = bbox[2] * bbox[3]
        anns.append(dict(
            id=ann_id_gen.get(),
            image_id=img_id,
            category_id=cat_id,
            bbox=bbox,
            area=area,
        ))
        
    img = dict(
        id=img_id,
        width=img_width,
        height=img_height,
        file_name=img_path,
    )
    
    return img, anns

def main():
    cats, img_paths = load_mat(os.path.join(root, 'train_list.mat'))
    train_ann_paths = list(map(lambda x: x.split('.')[0], img_paths))

    img_id_gen = IDGenerator()
    ann_id_gen = IDGenerator()
    
    imgs, anns = [], []
    for path in tqdm(train_ann_paths):
        xml_path = os.path.join(root, 'Annotations', path).replace('\\', '/')
        img, anns_ = load_xml(xml_path, cats, img_id_gen, ann_id_gen)
        imgs.append(img)
        for ann in anns_:
            anns.append(ann)
    
    with open('instances_train.json', 'w') as f:
        json.dump(dict(
            categories=cats,
            annotations=anns,
            images=imgs,
        ), f, indent=2)

    _, img_paths = load_mat(os.path.join(root, 'test_list.mat'))
    test_ann_paths = list(map(lambda x: x.split('.')[0], img_paths))

    img_id_gen = IDGenerator()
    ann_id_gen = IDGenerator()

    imgs, anns = [], []
    for path in tqdm(test_ann_paths):
        xml_path = os.path.join(root, 'Annotations', path).replace('\\', '/')
        img, anns_ = load_xml(xml_path, cats, img_id_gen, ann_id_gen)
        imgs.append(img)
        for ann in anns_:
            anns.append(ann)
    
    with open('instances_test.json', 'w') as f:
        json.dump(dict(
            categories=cats,
            annotations=anns,
            images=imgs,
        ), f, indent=2)

    
if __name__ == '__main__':
    main()