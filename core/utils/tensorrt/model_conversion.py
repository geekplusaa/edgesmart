# # -*- coding: UTF-8 -*-
# import torch
# import torch.onnx
# from tinynet import tinynet
# from conf import settings
# import os
#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input'], output_names=['output'], device='cpu'):
#     if not onnx_path.endswith('.onnx'):
#         print('Warning! The onnx model name is not correct,\
#                   please give a name that ends with \'.onnx\'!')
#         return 0
#
#     model = tinynet()
#     model.load_state_dict(torch.load(checkpoint))
#     model.eval()
#     # model.to(device)
#
#     torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names)
#     print("Exporting .pth model to onnx model has been successful!")
#
#
# class model_conversion(object):
#
#     def __init__(self):
#         pass
#
#
# if __name__ == "__main__":
#     names_pt_file_path = 'D:/ai/projects/ai-projects/flask-model/models/mask_signatrix_efficientdet_coco_best_epoch30.pth'
#     names_onnx_file_path = 'D:/ai/projects/ai-projects/flask-model/models/mask_signatrix_efficientdet_coco_best_epoch30.onnx'
#     database_pt_file_path = 'D:/ai/projects/facenet/person-name/facenet/database/database.pt'
#     database_onnx_file_path = 'D:/ai/projects/facenet/person-name/facenet/database/database.onnx'
#     input = torch.randn(1, 1, 640, 360)
#     object = model_conversion()
#
#     pth_to_onnx(input, names_pt_file_path, names_onnx_file_path)
