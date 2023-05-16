from torch.utils.data import Dataset
import os
import torch
import numpy as np
from models.orcalign_trimmed_ai_upper import pointnet2_part_seg_msg as segmentation_model



seg_classes = {'jaw': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]}
gum_classes = {'gum': [0, 1]}

seg_label_to_cat = {}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def parse_json_to_stl(json_stl):
    return json_stl


class PartRequestStl(Dataset):
    def __init__(self, data, npoints=25000, task="toothSegmentation"):
        self.npoints = npoints
        self.data = [np.array(data)]
        self.cat = {}
        self.classes = {}
        self.normal_channel = True
        for i in self.cat.keys():
            self.classes[i] = self.classes_original[i]

        if task == "toothSegmentation":
            self.seg_classes = seg_classes
        elif task == "trimGum":
            self.seg_classes = gum_classes
        else:
            raise ValueError('Task is not defined!')

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 25000

    def __getitem__(self, index):
        cls = np.array([0]).astype(np.int32)
        data = self.data[index]
        if not self.normal_channel:
            point_set = data[:, 0:3].copy()
        else:
            point_set = data[:, 0:6].copy()
        print("Length of data: ", len(point_set))
        point_set_original = data[:, 0:3]
        if not self.normal_channel:
            point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        else:
            point_set[:, 0:4] = pc_normalize(point_set[:, 0:4])

        import numpy.random as local_random
        local_random.seed(31337)

        choice = local_random.choice(len(point_set), self.npoints, replace=False)

        point_set_normal = point_set[choice, :]
        return point_set_normal, cls, point_set_original, choice

    def __len__(self):
        print('length of data:', len(self.data))
        return len(self.data)


def get_tooth_segmentation_vertices(data, model_dir, jaw):
    if jaw=="upper":
        npoints = 25000
    else:
        npoints = 20000
    TEST_DATASET = PartRequestStl(data, npoints, task="toothSegmentation")
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET)
    num_votes = 3
    num_classes = 1
    num_part = 17
    '''HYPER PARAMETER'''
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    '''MODEL LOADING'''
    MODEL = segmentation_model
    # print(MODEL)
    if torch.cuda.is_available():
        classifier = MODEL.get_model(num_part, normal_channel=True).cuda()
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth')
    else:
        classifier = MODEL.get_model(num_part, normal_channel=True)
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))

    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        seg_label_to_cat = {}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for points, label, ps, idx in (testDataLoader):
            batchsize, num_point, _ = points.size()
            cur_batch_size, points_count, _ = points.size()
            if torch.cuda.is_available():
                points, labels, ps = points.float().cuda(), label.long().cuda(), ps.float().cuda()
                vote_pool = torch.zeros(1, npoints, num_part).cuda()
            else:
                points, labels, ps = points.float(), label.long(), ps.float()
                vote_pool = torch.zeros(1, npoints, num_part)

            points = points.transpose(2, 1)

            for _ in range(num_votes):  # num_votes
                seg_pred, _ = classifier(points, to_categorical(labels, num_classes))
                vote_pool += seg_pred
            seg_pred = vote_pool / num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, points_count)).astype(np.int32)

            logits = cur_pred_val_logits[0, :, :]
            cur_pred_val[0, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

            point_index = idx.numpy()
        print('Model sent successfully ...')
        return cur_pred_val[0], point_index[0]


def get_trim_gum_vertices(data, jaw_class):
    npoints = 16384
    TEST_DATASET = PartRequestStl(data, npoints, task="trimGum")
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET)
    num_votes = 3
    num_classes = 1
    num_part = 2
    '''HYPER PARAMETER'''
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if jaw_class == 0:  # Gips lower class = 0
        model_dir = 'models/gibs_lower_jaw'
    elif jaw_class == 1:    # Gips upper class = 1
        model_dir = 'models/gibs_upper_jaw'
    elif jaw_class == 2:    # IO lower class = 2
        model_dir = 'models/gibs_upper_jaw'
    else:                   # IO upper class = 3
        model_dir = 'models/gibs_upper_jaw'

    '''MODEL LOADING'''
    MODEL = segmentation_model
    # print(MODEL)
    if torch.cuda.is_available():
        classifier = MODEL.get_model(num_part, normal_channel=True).cuda()
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth')
    else:
        classifier = MODEL.get_model(num_part, normal_channel=True)
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))

    classifier.load_state_dict(checkpoint['model_state_dict'])
    with torch.no_grad():
        seg_label_to_cat = {}
        for cat in gum_classes.keys():
            for label in gum_classes[cat]:
                seg_label_to_cat[label] = cat

        classifier = classifier.eval()

        for origin_points, label, ps, idx in (testDataLoader):
            batchsize, num_point, _ = origin_points.size()
            cur_batch_size, points_count, _ = origin_points.size()
            if torch.cuda.is_available():
                points, labels, ps = origin_points.float().cuda(), label.long().cuda(), ps.float().cuda()
                vote_pool = torch.zeros(1, npoints, num_part).cuda()
            else:
                points, labels, ps = origin_points.float(), label.long(), ps.float()
                vote_pool = torch.zeros(1, npoints, num_part)

            points = points.transpose(2, 1)
            for _ in range(num_votes):  # num_votes
                seg_pred, _ = classifier(points, to_categorical(labels, num_classes))
                vote_pool += seg_pred
            seg_pred = vote_pool / num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, points_count)).astype(np.int32)

            logits = cur_pred_val_logits[0, :, :]
            cur_pred_val[0, :] = np.argmax(logits[:, gum_classes[cat]], 1) + gum_classes[cat][0]

            point_index = idx.numpy()
            original_points = origin_points.numpy()
        print('Model sent successfully ...')
        return cur_pred_val[0], point_index[0], original_points[0]