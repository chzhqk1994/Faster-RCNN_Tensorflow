# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
import tensorflow as tf
import tensorflow.contrib.slim as slim
import time
import cv2
import pickle
import numpy as np
sys.path.append("../")

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.val_libs import voc_eval
from libs.box_utils import draw_box_in_img
from libs.label_name_dict.label_dict import LABEL_NAME_MAP, NAME_LABEL_MAP
import argparse
from help_utils import tools

EVAL_INTERVAL = 300 #60 seconds, for test
# EVAL_INTERVAL = 60 #60 seconds, for test

def calc_fscore(precision, recall, beta):
    return (1+beta**2)*precision*recall / (beta**2*precision + recall)

def eval_with_plac(img_dir, det_net, num_imgs, image_ext, draw_imgs, test_annotation_path):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not BGR
    img_batch = tf.cast(img_plac, tf.float32)

    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)

    detection_boxes, detection_scores, detection_category = det_net.build_whole_detection_network(
        input_img_batch=img_batch,
        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    global_step_tensor = slim.get_or_create_global_step()

    eval_result = []
    last_checkpoint_name = None

    while True:

        restorer, restore_ckpt = det_net.get_restorer()
        start_time = time.time()

        model_path = os.path.splitext(os.path.basename(restore_ckpt))[0]
        if model_path == None:
            print("Wait for available checkpoint")
        elif last_checkpoint_name == model_path:
            print("Already evaluated checkpoint {}, we will try evaluation in {} seconds".format(model_path,
                                                                                                 EVAL_INTERVAL))
            # continue
        else:
            print('Last ckpt was {}, new ckpt is {}'.format(last_checkpoint_name, model_path))
            last_checkpoint_name = model_path

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True

            with tf.Session(config=config) as sess:
                sess.run(init_op)
                sess.run(global_step_tensor.initializer)
                if not restorer is None:
                    restorer.restore(sess, restore_ckpt)
                    print('restore model')

                global_stepnp = tf.train.global_step(sess, global_step_tensor)
                print('#########################', global_stepnp)

                all_boxes = []
                imgs = os.listdir(img_dir)
                for i, a_img_name in enumerate(imgs):
                    a_img_name = a_img_name.split(image_ext)[0]
                    print('\n', a_img_name)

                    raw_img = cv2.imread(os.path.join(img_dir, a_img_name + image_ext))
                    raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

                    start = time.time()
                    resized_img, detected_boxes, detected_scores, detected_categories = \
                        sess.run(
                            [img_batch, detection_boxes, detection_scores, detection_category],
                            feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                        )
                    end = time.time()
                    # print("{} cost time : {} ".format(img_name, (end - start)))
                    if draw_imgs:
                        det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
                                                                       boxes=detected_boxes,
                                                                       labels=detected_categories,
                                                                       scores=detected_scores)
                        save_dir = os.path.join(cfgs.TEST_SAVE_PATH, cfgs.VERSION)
                        tools.mkdir(save_dir)
                        cv2.imwrite(save_dir + '/' + a_img_name + '.jpg',
                                    det_detections_h[:, :, ::-1])

                    xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                             detected_boxes[:, 2], detected_boxes[:, 3]

                    resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

                    xmin = xmin * raw_w / resized_w
                    xmax = xmax * raw_w / resized_w
                    ymin = ymin * raw_h / resized_h
                    ymax = ymax * raw_h / resized_h

                    boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
                    dets = np.hstack((detected_categories.reshape(-1, 1),
                                      detected_scores.reshape(-1, 1),
                                      boxes))
                    all_boxes.append(dets)

                    tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))

                fw1 = open(os.path.join(save_dir, 'detections.pkl'), 'wb')
                pickle.dump(all_boxes, fw1)

                with open(os.path.join(save_dir, 'detections.pkl'), 'rb') as f1:
                    all_boxes_h = pickle.load(f1, encoding='unicode')

                    print(10 * "###")
                    print(len(all_boxes_h))

                xmls = os.listdir(test_annotation_path)
                real_test_imgname_list = [i.split('.xml')[0] for i in xmls]

                print(10 * "**")
                print('horizon eval:')
                # print(len(all_boxes_h), len(all_boxes_r))
                # print(len(real_test_imgname_list))
                mAP_h, recall_h, precision_h, total_mAP_h, total_recall_h, total_precision_h = voc_eval.voc_evaluate_detections(
                    all_boxes=all_boxes_h,
                    test_imgid_list=real_test_imgname_list,
                    test_annotation_path=test_annotation_path)
                print('mAP_h: ', mAP_h)
                print('mRecall_h:', recall_h)
                print('mPrecision_h:', precision_h)
                print('total_mAP_h: ', total_mAP_h)
                print('total_recall_h_list:', total_recall_h)
                print('total_precision_h_list:', total_precision_h)

                f1score_h_check = (1 + 1 ** 2) * precision_h * recall_h / (1 ** 2 * precision_h + recall_h)
                f1score_h = calc_fscore(precision_h, recall_h, 1)

                summary_path = os.path.join(cfgs.SUMMARY_PATH, cfgs.VERSION + '/eval_0')
                tools.mkdir(summary_path)

                summary_writer = tf.summary.FileWriter(summary_path, graph=sess.graph)

                mAP_h_summ = tf.Summary()
                mAP_h_summ.value.add(tag='EVAL_Global/mAP_h', simple_value=mAP_h)
                summary_writer.add_summary(mAP_h_summ, global_stepnp)

                mRecall_h_summ = tf.Summary()
                mRecall_h_summ.value.add(tag='EVAL_Global/Recall_h', simple_value=recall_h)
                summary_writer.add_summary(mRecall_h_summ, global_stepnp)

                mPrecision_h_summ = tf.Summary()
                mPrecision_h_summ.value.add(tag='EVAL_Global/Precision_h', simple_value=precision_h)
                summary_writer.add_summary(mPrecision_h_summ, global_stepnp)

                mF1Score_h_summ = tf.Summary()
                mF1Score_h_summ.value.add(tag='EVAL_Global/F1Score_h', simple_value=f1score_h)
                summary_writer.add_summary(mF1Score_h_summ, global_stepnp)

                mAP_h_class_dict = {}
                recall_h_class_dict = {}
                precision_h_class_dict = {}
                f1score_h_class_dict = {}

                label_list = list(NAME_LABEL_MAP.keys())
                label_list.remove('back_ground')

                for cls in label_list:
                    mAP_h_class_dict["cls_%s_mAP_h_summ" % cls] = tf.Summary()
                    recall_h_class_dict["cls_%s_recall_h_summ" % cls] = tf.Summary()
                    precision_h_class_dict["cls_%s_precision_h_summ" % cls] = tf.Summary()
                    f1score_h_class_dict["cls_%s_f1score_h_summ" % cls] = tf.Summary()

                for cls in label_list:
                    mAP_h_class_dict["cls_%s_mAP_h_summ" % cls].value.add(tag='EVAL_Class_mAP/{}_mAP_h'.format(cls),
                                                                          simple_value=total_mAP_h[cls])
                    recall_h_class_dict["cls_%s_recall_h_summ" % cls].value.add(
                        tag='EVAL_Class_recall/{}_recall_h'.format(cls), simple_value=total_recall_h[cls])
                    precision_h_class_dict["cls_%s_precision_h_summ" % cls].value.add(
                        tag='EVAL_Class_precision/{}_precision_h'.format(cls), simple_value=total_precision_h[cls])

                    f1score_h_cls = calc_fscore(total_precision_h[cls], total_recall_h[cls], 1)
                    f1score_h_class_dict["cls_%s_f1score_h_summ" % cls].value.add(
                        tag='EVAL_Class_f1score/{}_f1score_h'.format(cls), simple_value=f1score_h_cls)

                for cls in label_list:
                    summary_writer.add_summary(mAP_h_class_dict["cls_%s_mAP_h_summ" % cls], global_stepnp)
                    summary_writer.add_summary(recall_h_class_dict["cls_%s_recall_h_summ" % cls], global_stepnp)
                    summary_writer.add_summary(precision_h_class_dict["cls_%s_precision_h_summ" % cls], global_stepnp)
                    summary_writer.add_summary(f1score_h_class_dict["cls_%s_f1score_h_summ" % cls], global_stepnp)

                summary_writer.flush()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_ckpt = os.path.join(save_dir, 'voc_' + str(global_stepnp) + 'model.ckpt')
            # saver.save(sess, save_ckpt)
            print(' weights had been saved')

            time_to_next_eval = start_time + EVAL_INTERVAL - time.time()
            if time_to_next_eval > 0:
                time.sleep(time_to_next_eval)


def eval(num_imgs, img_dir, image_ext, test_annotation_path):

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)

    eval_with_plac(img_dir=img_dir, det_net=faster_rcnn, num_imgs=num_imgs, image_ext=image_ext, draw_imgs=True, test_annotation_path=test_annotation_path)

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--img_dir', dest='img_dir',
                        help='images path',
                        default='/home/qisens/Faster-RCNN_Tensorflow/tools/eval_images_area123/JPEGImages', type=str)
    parser.add_argument('--image_ext', dest='image_ext',
                        help='image format',
                        default='.png', type=str)
    parser.add_argument('--test_annotation_path', dest='test_annotation_path',
                        help='test annotate path',
                        default='/home/qisens/Faster-RCNN_Tensorflow/tools/eval_images_area123/annotations', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='2', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    eval(np.inf, args.img_dir, args.image_ext, args.test_annotation_path)















