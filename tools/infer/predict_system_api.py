from flask import Flask, request, jsonify, abort
import base64
import requests
from datetime import datetime
import subprocess
import os
import sys
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import cv2
import tools.infer.predict_det as predict_det
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_cls as predict_cls
import copy
import numpy as np
import math
import time
import tools.infer.utility as utility
from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from PIL import Image
from tools.infer.utility import draw_ocr
from tools.infer.utility import draw_ocr_box_txt
from ppocr.utils.logging import get_logger

app = Flask(__name__)

logger = get_logger()

class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            print(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        print("dt_boxes num : {}, elapse : {}".format(len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            print("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))
        rec_res, elapse = self.text_recognizer(img_crop_list)
        print("rec_res num  : {}, elapse : {}".format(len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        return dt_boxes, rec_res

def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def do_predict(args):
    """
    :param args:
    :return: 返回(image_file, newdt, rec_res)组成的列表，image_file是图片，newdt是文字的box坐标，rec_res是识别结果和置信度组成的元祖
    """
    image_file_list = get_image_file_list(args.image_dir)
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    #把所有的图片的名字，bbox，识别结果放在一个tuple里面返回
    images_result = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        # dt_boxes 是一个列表，列表中每个元素时一个bbox坐标，格式是，每个点是x,y，每个点的位置是[左上角点，右上角点,右下角点，左下角点] [[171.  93.], [626.  93.], [626. 139.], [171. 139.]]
        # rec_res 是识别结果和置信度的元祖组成的列表, 其中的一个元素是['为你定制元气美肌', 0.9992783]
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))

        drop_score = 0.5
        dt_num = len(dt_boxes)
        for dno in range(dt_num):
            text, score = rec_res[dno]
            if score >= drop_score:
                text_str = "%s, %.3f" % (text, score)
                print(text_str)
        if is_visualize:
            # 是否可视化
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)
            draw_img_save = "./inference_results/"
            if not os.path.exists(draw_img_save):
                os.makedirs(draw_img_save)
            cv2.imwrite(
                os.path.join(draw_img_save, os.path.basename(image_file)),
                draw_img[:, :, ::-1])
            print("The visualized image saved in {}".format(
                os.path.join(draw_img_save, os.path.basename(image_file))))
        #dt的numpy改成列表格式
        newdt = [i.tolist() for i in dt_boxes]
        # 把置信度float32改成字符串格式，方便以后json dumps
        rec_res = [[rec[0],str(rec[1])] for rec in rec_res]
        images_result.append((image_file, newdt, rec_res))
    return images_result


@app.route("/api/base64", methods=['POST'])
def b64():
    """
    接收用户调用paddle ocr api接口
    返回json格式的预测结果
    :param contents, 提供的文件内容，是一个列表结构, base64格式
    :return: json格式  [{name:图片名称,label: 分类结果，content: 识别内容},{name:图片名称,label: 分类结果，content: 识别内容}]
    """
    save_path = '/tmp/'
    jsonres = request.get_json()
    #images是图片的base64格式
    images= jsonres.get('images', None)
    #图片的名字列表
    names= jsonres.get('names', None)
    #创建一个目录为这次请求，图片保存到这个目录下，识别结果也是从这里拿到
    image_dir = os.path.join(save_path,datetime.now().strftime('%Y%m%d%H%M%S%f'))
    if not os.path.exists(image_dir):
        os.mkdir(image_dir)
    for name,image in zip(names,images):
        name = os.path.basename(name)
        file_timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        image_path = os.path.join(image_dir, "ocr{}_{}".format(file_timestamp,name))
        #解压成base64, 保存到本地
        file = base64.b64decode(image)
        with open(image_path, "wb") as f:
            f.write(file)
    args = parse_args()
    #识别图片的路径
    args.image_dir = image_dir
    args.det_model_dir = "inference/ch_ppocr_mobile_v2.0_det_infer/"
    args.rec_model_dir = "inference/ch_ppocr_mobile_v2.0_rec_infer/"
    args.cls_model_dir = "inference/ch_ppocr_mobile_v2.0_cls_infer/"
    args.use_angle_cls = True
    args.use_space_char = True
    #不画出结果图
    # args.is_visualize = False
    images_result = do_predict(args)
    results = {"content": images_result}
    return jsonify(results)


@app.route("/api/path", methods=['POST'])
def path():
    """
    传过来给定图片的路径即可，需要绝对路径
    :return:
    :rtype:
    """
    jsonres = request.get_json()
    #images是图片的路径
    image_dir= jsonres.get('images', None)
    #识别图片的路径
    image_file_list = get_image_file_list(image_dir)
    #把所有的图片的名字，bbox，识别结果放在一个tuple里面返回
    images_result = []
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()
        # dt_boxes 是一个列表，列表中每个元素时一个bbox坐标，格式是，每个点是x,y，每个点的位置是[左上角点，右上角点,右下角点，左下角点] [[171.  93.], [626.  93.], [626. 139.], [171. 139.]]
        # rec_res 是识别结果和置信度的元祖组成的列表, 其中的一个元素是['为你定制元气美肌', 0.9992783]
        dt_boxes, rec_res = text_sys(img)
        elapse = time.time() - starttime
        print("Predict time of %s: %.3fs" % (image_file, elapse))
        # drop_score = 0.5
        # dt_num = len(dt_boxes)
        # for dno in range(dt_num):
        #     text, score = rec_res[dno]
        #     if score >= drop_score:
        #         text_str = "%s, %.3f" % (text, score)
        #         print(text_str)
            # dt的numpy改成列表格式
        every_res = []
        for rec, dt in zip(rec_res,dt_boxes):
            dt = dt.tolist()
            one_res = {
                "words": rec[0],
                "confidence": str(rec[1]),
                "left_top": dt[0],
                "right_top": dt[1],
                "right_bottom":dt[2],
                "left_bottom":dt[3],
            }
            every_res.append(one_res)
        one_data = {
            "image_name": image_file,
            "ocr_result": every_res
        }
        images_result.append(one_data)
    return jsonify(images_result)


if __name__ == "__main__":
    args = utility.parse_args()
    args.det_model_dir = "inference/ch_ppocr_mobile_v2.0_det_infer"
    args.rec_model_dir = "inference/ch_ppocr_mobile_v2.0_rec_infer"
    args.cls_model_dir = "inference/ch_ppocr_mobile_v2.0_cls_infer"
    args.use_angle_cls = True
    args.use_space_char = True
    text_sys = TextSystem(args)
    app.run(host='0.0.0.0', port=6688, debug=False, threaded=True)
