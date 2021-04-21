import os
import sys

import torch

from eval.eval_angles import eval_angles


def tmpModel(path="/home/leo/samba107/lpy/LightCNN_29Layers_V2_checkpoint.pth.tar"):
    from tmp.light_cnn import LightCNN_29Layers_v2
    model = LightCNN_29Layers_v2(num_classes=80013)
    if torch.cuda.is_available() is True:
        model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(path)['state_dict'])
    model.eval()
    return model


def assertPath(pathes):
    pks = []
    for path in pathes:
        assert (path[-1] != "/")
        pks.append(os.path.basename(path) + ".pk")
    return pks


def GetProbeAndGallery(
        # name="MegaFace",
        name="JieBang",
):
    if name == "JieBang":
        gallery_pathes = [
            "/home/leo/samba107/lpy/persons/4000Noise/new_projection_new4000/4000Noise_hierarchy_front_aligned",
            "/home/leo/samba107/lpy/persons/pers_306_raw_aligned_centerfacePose"
        ]
        probe_root_path = "/home/leo/samba107/lpy/persons/probe_9306_raw_aligned_centerfacePose"
    elif name == "MegaFace":
        gallery_pathes = [
            "/home/leo/samba107/lpy/persons/megaFaceCMC/mega_gallery",
            "/home/leo/samba107/lpy/persons/megaFaceCMC/mega_extra_gallery10W"
        ]
        probe_root_path = "/home/leo/samba107/lpy/persons/megaFaceCMC/mega_probe"

    return probe_root_path, gallery_pathes


def modelFactory():
    from evalDataSets.eval_angles import eval_angles
    from RealTimeEval.senseTimeCos.impl_sTCos_face_public import SENTIMEModel
    from RealTimeEval.wiseSoftL29v2block5.light29B5.extract_features29_b import GLight29v2B5Load
    from thirdPartyModel.FaceX.AttentionNets import FaceXModel
    from thirdPartyModel.InsightFace.iresnet import IResNet100Glint36KCosFaceFC, IResNet50Glint36KCosFaceFC

    torch.multiprocessing.set_start_method('spawn')

    model = GLight29v2B5Load()
    # model = SENTIMEModel()
    # model = FaceXModel()
    # model = tmpModel()
    # model = IResNet100Glint36KCosFaceFC()
    # model = IResNet50Glint36KCosFaceFC()

    return model


if __name__ == '__main__':
    model = modelFactory()

    probe_root_path, gallery_pathes = GetProbeAndGallery()
    cacheOfGallery = assertPath(gallery_pathes)
    assertPath([probe_root_path])

    acc = eval_angles(model, gallery_pathes, cacheOfGallery, probe_root_path, -1, 16,
                      os.path.basename(probe_root_path) + ".pk", mode="ALL", visualise=False)


def EvalJieBang(model, rank, epoch, num_update):
    if rank is 0 and num_update > 0 and num_update % 2000 == 0:
        probe_root_path, gallery_pathes = GetProbeAndGallery()
        cacheOfGallery = assertPath(gallery_pathes)
        assertPath([probe_root_path])

        acc = eval_angles(model, gallery_pathes, cacheOfGallery, probe_root_path, epoch, 8,
                          os.path.basename(probe_root_path) + ".pk", mode="ALL", visualise=False)
        return acc
    return None

# JieBang
# 9459  Epoch:-1 default acc: 0.9468 , avgTop1Scores:0.8604 max:0.9669 min:0.2514 median:0.9086 60th:0.9545 80th:0.7999
# senseTime Epoch:-1 default acc: 0.9514 , avgTop1Scores:0.8644 max:0.9638 min:0.2194 median:0.9104 60th:0.9536 80th:0.8190
# FaceX Epoch:-1 default acc: 0.4755 , avgTop1Scores:0.8502 max:0.9622 min:0.3125 median:0.9025 60th:0.9706 80th:0.7543
# L29v2 Epoch:-1 default acc: 0.6669 , avgTop1Scores:0.9199 max:0.9789 min:0.5729 median:0.9245 60th:0.9995 80th:0.9829
# InsightFace Iresnet cosFace f16 FC   Epoch:-1 default acc: 0.8240 , avgTop1Scores:0.8748 max:0.9659 min:0.3488 median:0.9098 60th:0.9843 80th:0.8422

# MegaFace
# 9459  Epoch:-1 default acc: 0.8501 , avgTop1Scores:0.9357 max:1.0000 min:0.7153 median:0.9360 60th:1.0000 80th:0.9980
# senseTime Epoch:-1 default acc: 0.8890 , avgTop1Scores:0.9350 max:1.0000 min:0.6808 median:0.9368 60th:1.0000 80th:0.9948
# InsightFace Glint360K Epoch:-1 default acc: 0.9348 , avgTop1Scores:0.9330 max:1.0000 min:0.5104 median:0.9391 60th:0.9984 80th:0.9798
