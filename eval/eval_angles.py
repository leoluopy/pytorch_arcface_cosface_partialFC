import pickle
import time

import jpeg4py
import torch
import os, sys
import cv2
from tqdm import tqdm
import numpy as np
from os import cpu_count
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def AllTransForms():
    transform_insightFace_Iresnet = transforms.Compose(
        [transforms.ToPILImage(),
         # transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
         ])  # input RGB image
    return transform_insightFace_Iresnet


def torch_default_loader_useForInsightFace(bgrImg224, meanStd=False,
                                           # For L29V2
                                           # dim=224,
                                           # transpose=True,
                                           # transforms=None,

                                           # For Insight Face
                                           dim=112,
                                           transpose=False,
                                           transforms=AllTransForms()
                                           ):
    if bgrImg224.shape[0] != dim:
        bgrImg224 = cv2.resize(bgrImg224, (dim, dim))
        # bgrImg224 = cv2.cvtColor(bgrImg224, cv2.COLOR_BGR2GRAY)
    if transforms is not None:
        bgrImg224 = cv2.cvtColor(bgrImg224, cv2.COLOR_BGR2RGB)
        bgrImg224 = transforms(bgrImg224)
    if torch.cuda.is_available():
        img = torch.tensor(bgrImg224).cuda().float()
    else:
        img = torch.tensor(bgrImg224).float()

    if transpose is True:
        img = img.transpose(2, 0).transpose(1, 2)
    # img = img / 255.0
    if meanStd:
        img = (img - 127.5) / 128.0
    return img


def angle2FileStr(yaw_predicted, pitch_predicted, roll_predicted):
    if abs(pitch_predicted) <= 15:
        pitch_str = "00"
    else:
        pitch_str = "20"
    if pitch_predicted < 0 and pitch_str != "00":
        pitch_str = "-" + pitch_str

    if abs(yaw_predicted) <= 15:
        yaw_str = "00"
    elif abs(yaw_predicted) > 15 and abs(yaw_predicted) <= 45:
        yaw_str = "30"
    else:
        yaw_str = "60"
    if yaw_predicted < 0 and yaw_str != "00":
        yaw_str = "-" + yaw_str

    choose_file_str = pitch_str + "_" + yaw_str + "_" + "00.jpg"
    return choose_file_str


def ChangeScore__(score):
    if (score > 0.2 and score <= 0.3):
        score = (score - 0.2) * 3 + 0.2
    elif (score > 0.3 and score <= 0.5):
        score = (score - 0.3) * 2 + 0.5
    elif (score > 0.5):
        score = (score - 0.5) * 0.2 + 0.9
    elif (score < 0.0):
        score = 0.0
    return score


def ChangeScoreTorchConcurrent(batch_scores):
    batch_scores = torch.where(batch_scores > 0, batch_scores, torch.zeros_like(batch_scores))
    batch_scores = torch.where(batch_scores > 0.5, (batch_scores - 0.5) * 0.2 + 0.9, batch_scores)
    masks = torch.ones_like(batch_scores) * -100.0

    selected_2_3 = torch.where(batch_scores > 0.2, batch_scores, masks)
    selected_2_3 = torch.where(selected_2_3 <= 0.3, selected_2_3, masks)
    selected_2_3 = (selected_2_3 - 0.2) * 3 + 0.2
    selected_2_3 = torch.where(selected_2_3 > 0, selected_2_3, masks)

    selected_3_5 = torch.where(batch_scores > 0.3, batch_scores, masks)
    selected_3_5 = torch.where(selected_3_5 <= 0.5, selected_3_5, masks)
    selected_3_5 = (selected_3_5 - 0.3) * 2 + 0.5
    selected_3_5 = torch.where(selected_3_5 > 0, selected_3_5, masks)

    batch_scores = torch.where(selected_2_3 != -100.0, selected_2_3, batch_scores)
    batch_scores = torch.where(selected_3_5 != -100.0, selected_3_5, batch_scores)

    return batch_scores


def acc_ids(gallery_ids, gallery_pathes, gt_ids, indices, scores, pathes, visualise=False, timeElapse=25, maxTopN=1):
    b_cnt = 0
    right_cnt = 0
    right_scores = []
    for b_idxs in indices:
        gt = gt_ids[b_cnt]
        predict_ids = []
        predict_ids_str = ""
        predict_scores = []
        predict_scores_str = ""
        for i in range(maxTopN):
            predict_ids.append(gallery_ids[b_idxs[i]])
            predict_ids_str += str(gallery_ids[b_idxs[i]]) + ","
            predict_scores.append(float(scores[b_cnt][i]))
            predict_scores_str += "{:.2f}".format(float(scores[b_cnt][i])) + ","

        predict_id = predict_ids[0]
        score = predict_scores[0]
        if visualise:
            img_gallery = cv2.imread(gallery_pathes[b_idxs[0]])
            # print("gallery: {}".format(gallery_pathes[b_idxs[0]]))
            img_inference = cv2.imread(pathes[b_cnt])
            # print("inf: {}".format(pathes[b_cnt]))
            cv2.putText(img_gallery, "GT:" + str(gt), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.putText(img_inference, "predicted:" + str(predict_id), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1,
                        (255, 255, 0), 1)
            cv2.putText(img_inference, "{:.4f}".format(score), (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cat_im = np.concatenate((img_gallery, img_inference), 1)
            cv2.putText(cat_im, "TOPN:" + predict_ids_str, (20, 200), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
            cv2.putText(cat_im, "TOPN:" + predict_scores_str, (20, 220), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
        if gt == predict_id:
            right_cnt += 1
            right_scores.append(score)
            if visualise:
                cv2.imshow("positive", cat_im)
                key = cv2.waitKey(timeElapse)
                if key == 99:
                    cv2.waitKey(0)
        else:
            if visualise:
                cv2.imshow("negative", cat_im)
                key = cv2.waitKey(timeElapse)
                if key == 99:
                    cv2.waitKey(0)
        b_cnt += 1

    return float(right_cnt), float(indices.shape[0]), right_scores


class IDsSetWithAngle(Dataset):
    def getAngles(self):
        for i in range(len(self.yaws)):
            pose_str = angle2FileStr(self.yaws[i], self.pitches[i], self.rolls[i]).split(".")[0]
            self.pose_strs.append(pose_str)

    def setAngleMode(self, mode):
        self.angle_mode = mode

    def initAngleModeFiles(self):
        self.angle_filterd_files.clear()
        if self.angle_mode != "ALL":
            assert (len(self.pose_strs) == len(self.file_paths))
            for i in range(len(self.file_paths)):
                if self.pose_strs[i] == self.angle_mode:
                    self.angle_filterd_files.append(self.file_paths[i])
                    self.angle_filterd_ids.append(self.file_IDs[i])
            print(" DataSetAngleMode Set : {} , now len: {}".format(self.angle_mode, len(self.angle_filterd_files)))
            sys.stdout.flush()
        else:
            self.angle_filterd_files = self.file_paths
            self.angle_filterd_ids = self.file_IDs

    def __init__(self, root_dirs, cache_file, angle_mode="00_00_00"):
        super(Dataset, self).__init__()
        self.root_dirs = root_dirs
        self.file_paths = []
        self.file_IDs = []
        self.AllIDs = []
        self.ID2LabelDict = {}
        self.pose_strs = []
        self.angle_mode = angle_mode
        self.angle_filterd_files = []
        self.angle_filterd_ids = []
        self.yaws, self.pitches, self.rolls = [], [], []

        start = time.time()
        if os.path.exists(cache_file) is False:
            for root_dir in tqdm(self.root_dirs):
                for dir in os.listdir(root_dir):
                    if os.path.isdir(os.path.join(root_dir, dir)) is False:
                        continue
                    else:
                        self.AllIDs.append(dir)
                        for file in os.listdir(os.path.join(root_dir, dir)):
                            if os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg', '.bmp', '.JPG', '.PNG', '.JPEG',
                                                             '.BMP']:
                                self.file_paths.append(os.path.join(root_dir, dir, file))
                                self.file_IDs.append(dir)
            label = 0
            for ID in self.AllIDs:
                self.ID2LabelDict[ID] = label
                label += 1
            if angle_mode != "ALL":
                print(" COMPUTING POSE OF {}".format(self.root_dirs))
                self.yaws, self.pitches, self.rolls = fullGetPosesOfPathes(self.file_paths)
                self.getAngles()
            print(" \n IDsSetWithAngle LOAD DONE size: {}".format(len(self.file_paths)))
            with open(cache_file, "wb") as f:
                pickle.dump([
                    self.file_paths,
                    self.file_IDs,
                    self.AllIDs,
                    self.ID2LabelDict,
                    self.pose_strs,
                    self.yaws, self.pitches, self.rolls,
                ], f)
        else:
            with open(cache_file, "rb") as f:
                self.file_paths, self.file_IDs, self.AllIDs, self.ID2LabelDict, self.pose_strs, \
                self.yaws, self.pitches, self.rolls = pickle.load(f)
            print("IDsSetWithAngle set loaded from cache pairs len: {}".format(len(self.file_paths)))

        torch.cuda.empty_cache()
        print("IDsSetWithAngle loading pairs cost:{}".format(time.time() - start))
        sys.stdout.flush()

    def __getitem__(self, idx):
        image = cv2.imread(self.angle_filterd_files[idx])
        # image = jpeg4py.JPEG(self.angle_filterd_files[idx]).decode()
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        file_ID = self.angle_filterd_ids[idx]
        label = self.ID2LabelDict[file_ID]
        path = self.angle_filterd_files[idx]
        return torch_default_loader_useForInsightFace(image), label, file_ID, path

    def __len__(self):
        return len(self.angle_filterd_files)


def load_angle_gallery(model, gallery_root, eval_batch_size, cache_file, angle_mode):
    gallery_files, gallery_ids, gallery_feats = [], [], {}

    data_set = IDsSetWithAngle(root_dirs=[gallery_root], cache_file=cache_file, angle_mode=angle_mode)
    data_set.setAngleMode(angle_mode)
    data_set.initAngleModeFiles()
    # gallery_loader = DataLoader(data_set, batch_size=eval_batch_size, shuffle=False, num_workers=int(cpu_count() / 2))
    gallery_loader = DataLoader(data_set, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    print(" LOADING ANGLE GALLERY FEAT : {}".format(gallery_root))
    for (images, labels, file_ids, pathes) in tqdm(gallery_loader):
        images = images.cuda()
        labels = labels.cuda()
        feat = model(images)
        feat = feat / torch.norm(feat, dim=1, keepdim=True, p=2)
        if isinstance(gallery_feats, dict):
            gallery_feats = feat.detach().cpu()
        else:
            gallery_feats = torch.cat((gallery_feats, feat.detach().cpu()), 0)
        gallery_files.extend(list(pathes))
        gallery_ids.extend(list(file_ids))

    return gallery_files, gallery_ids, gallery_feats


def eval_angles(model, gallery_paths, cache_files, probe_root_path, epoch, eval_batch_size, cache_file_probe,
                mode="00_00_00", TAG="default", visualise=False):
    torch.cuda.empty_cache()
    model.eval()

    gallery_files, gallery_ids, gallery_feats_all = {}, {}, {}
    for i in range(len(gallery_paths)):
        gallery_files_one, gallery_ids_one, gallery_feats_one = load_angle_gallery(
            model, gallery_paths[i], eval_batch_size, cache_files[i], mode)
        if isinstance(gallery_files, dict):
            gallery_files, gallery_ids, gallery_feats_all = gallery_files_one, gallery_ids_one, gallery_feats_one
        else:
            gallery_files.extend(gallery_files_one)
            gallery_ids.extend(gallery_ids_one)
            gallery_feats_all = torch.cat((gallery_feats_all, gallery_feats_one), 0)

    rightCnt = 0.
    evalCnt = 0.
    whole_top1Scores = []

    data_set = IDsSetWithAngle(root_dirs=[probe_root_path], cache_file=cache_file_probe, angle_mode=mode)
    data_set.setAngleMode(mode)
    data_set.initAngleModeFiles()
    # probe_loader = DataLoader(data_set, batch_size=eval_batch_size, shuffle=False, num_workers=int(cpu_count() / 2))
    probe_loader = DataLoader(data_set, batch_size=eval_batch_size, shuffle=False, num_workers=0)

    print("         ##### START EVAL")
    for (images, labels, file_ids, pathes) in tqdm(probe_loader):
        images = images.cuda()
        labels = labels.cuda()

        feat_backbone = model(images)

        feat_backbone = feat_backbone / torch.norm(feat_backbone, dim=1, keepdim=True, p=2)
        feat_backbone = feat_backbone.detach().cpu()
        scores_backbone = feat_backbone @ gallery_feats_all.t()
        scores = ChangeScoreTorchConcurrent(scores_backbone)

        # scores = scores_backbone
        # [ b , gallery_featsNum 0,1,2 .... N ]
        # [ b , gallery_IDs id0, id1, id2 .... idN ]
        useTopk = 9
        topkscores, indices = scores.topk(useTopk, dim=1, largest=True, sorted=True)
        b_rightCnt, b_evalCnt, top1Scores = acc_ids(gallery_ids, gallery_files, list(file_ids), indices, topkscores,
                                                    pathes, maxTopN=useTopk, visualise=False)
        rightCnt += b_rightCnt
        evalCnt += b_evalCnt
        whole_top1Scores.extend(top1Scores)
        # visualise(images, gallery_pathes, indices)
    if len(whole_top1Scores) == 0:
        whole_top1Scores.append(0.0)
    print(
        " \n  Epoch:{} {} acc: {:.4f} , avgTop1Scores:{:.4f} max:{:.4f} min:{:.4f} median:{:.4f} 60th:{:.4f} 80th:{:.4f}".format(
            epoch, TAG,
            rightCnt / evalCnt,
            np.array(whole_top1Scores).mean(),
            np.array(whole_top1Scores).max(),
            np.array(whole_top1Scores).min(),
            np.median(np.array(whole_top1Scores)),
            (np.array(whole_top1Scores) > 0.6).sum() / len(whole_top1Scores),
            (np.array(whole_top1Scores) > 0.8).sum() / len(whole_top1Scores)))
    sys.stdout.flush()
    torch.cuda.empty_cache()

    return rightCnt / evalCnt, 1.0


def caseIterateAngleMode(data_set, mode):
    data_set.setAngleMode(mode)
    data_set.initAngleModeFiles()
    data_loader = DataLoader(data_set, batch_size=2, shuffle=False, num_workers=0)
    for (images, labels, file_ids, pathes) in tqdm(data_loader):
        image0 = images[0].transpose(0, 2).transpose(0, 1).detach().cpu().numpy().astype(np.uint8)
        print(labels[0])
        print(file_ids[0])
        print(pathes[0])
        cv2.imshow(mode, image0)
        cv2.waitKey(0)


def caseIDsSetWithAngle():
    root_dirs = [
        '/home/leo/workspace/data_set/K12_eval/negative/testP',
        "/home/leo/workspace/data_set/K12_eval/negative/top2_5Merge_aligned/",
        # "/home/leo/workspace/data_set/K12_eval/negative/20201201-20201204_aligned/lower_than_75",
        # "/home/leo/workspace/data_set/K12_eval/negative/20201214-20201223_aligned/lower_than_75",
    ]
    data_set = IDsSetWithAngle(root_dirs=root_dirs,
                               cache_file="tmpcache.pk")
    caseIterateAngleMode(data_set, "00_00_00")
    caseIterateAngleMode(data_set, "20_00_00")
    caseIterateAngleMode(data_set, "-20_00_00")

    caseIterateAngleMode(data_set, "00_30_00")
    caseIterateAngleMode(data_set, "20_30_00")
    caseIterateAngleMode(data_set, "-20_30_00")

    caseIterateAngleMode(data_set, "00_-30_00")
    caseIterateAngleMode(data_set, "20_-30_00")
    caseIterateAngleMode(data_set, "-20_-30_00")

    caseIterateAngleMode(data_set, "00_60_00")
    caseIterateAngleMode(data_set, "20_60_00")
    caseIterateAngleMode(data_set, "-20_60_00")
    caseIterateAngleMode(data_set, "00_-60_00")

    caseIterateAngleMode(data_set, "20_-60_00")
    caseIterateAngleMode(data_set, "-20_-60_00")


if __name__ == '__main__':
    caseIDsSetWithAngle()
