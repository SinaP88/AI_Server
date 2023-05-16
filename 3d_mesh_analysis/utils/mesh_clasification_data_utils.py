from torch.utils.data import Dataset
import torch
import numpy as np
from models.EnhancedBinaryJawClassification import pointnet2_cls_msg as jaw_classifier

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:-
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point

class ClassifierDataLoader(Dataset):
    def __init__(self, data, process_data=False, npoints=16384):
        self.data = np.expand_dims(data, axis=0)
        self.npoints = npoints
        self.process_data = process_data
        self.uniform = False
        self.use_normals = False
        self.num_category = 2
        self.cat = ["gibsMandibular", "gibsMaxillary"]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        data = self.data[index]

        point_set = np.array(data).astype(np.float32)
        if self.uniform:
            point_set = farthest_point_sample(point_set, self.npoints)
        else:
            point_set = point_set[0:self.npoints, :]

        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set



def predictJawClass(data, model_dir, num_class=2, vote_num=1):

    test_dataset = ClassifierDataLoader(data, process_data=False, npoints=16384)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    '''MODEL LOADING'''
    num_class = num_class
    MODEL = jaw_classifier
    # print(MODEL)

    if torch.cuda.is_available():
        classifier = MODEL.get_model(num_class, normal_channel=False).cuda()
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth')
    else:
        classifier = MODEL.get_model(num_class, normal_channel=False)
        checkpoint = torch.load(str(model_dir) + '/checkpoints/best_model.pth', map_location=torch.device('cpu'))

    classifier.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        classifier = classifier.eval()
        predictions = []
        for points in testDataLoader:
            if torch.cuda.is_available():
                points = points.transpose(2, 1).cuda()
            else:
                points = points.transpose(2, 1)

            for _ in range(vote_num):
                pred, _ = classifier(points)
            predictions.append(np.argmax(pred.cpu().detach().numpy()))

    return predictions
