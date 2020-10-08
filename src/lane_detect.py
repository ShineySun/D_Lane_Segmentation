# python3 inference_unet.py --netType Unet --GPUs 0 --batchSize 1

import sys
#print(sys.path)
#sys.path.append('/lib/python3.7/site-packages')
import opts
import math
import importlib
import os
from operator import itemgetter

import _init_paths
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D

import cv2
import numpy as np
from PIL import Image
from warper import Warper
from slidewindow import SlideWindow

import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def normalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] - mean[0]) / std[0]
    ipt[:][:][1] = (ipt[:][:][1] - mean[1]) / std[1]
    ipt[:][:][2] = (ipt[:][:][2] - mean[2]) / std[2]
    return ipt

def unNormalize(ipt, mean, std):
    ipt[:][:][0] = (ipt[:][:][0] * std[0]) + mean[0]
    ipt[:][:][1] = (ipt[:][:][1] * std[1]) + mean[1]
    ipt[:][:][2] = (ipt[:][:][2] * std[2]) + mean[2]
    return ipt

def preprocess_img(im):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = np.asarray(im)
    im = normalize(im, mean, std)
    im = np.transpose(im, (2, 0, 1))
    return im

def postprocess_img(im):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = np.transpose(im, (1, 2, 0))
    im = unNormalize(im, mean, std)
    return im

def dist_calc(line):
    x1, y1, x2, y2 = line[0]

    dist = (x1-x2)**2+(y1-y2)**2

    return np.sqrt(dist)

def vanishing_point_detector(rgb_img, line_img):
    # get image size
    (img_size_x, img_size_y) = rgb_img.shape[:2]

    van_img = rgb_img.copy()

    # dist_threshold
    dist_threshold = 15.

    print("* size_x : {}    * size_y : {}".format(img_size_x, img_size_y))

    real_lines = []

    #canny = cv2.Canny(line_img, 100,200)

    fld = cv2.ximgproc.createFastLineDetector()

    lines = fld.detect(line_img)

    if lines is None:
        return None

    # hough transform
    #lines = cv2.HoughLinesP(line_img, 1, np.pi/180., 150)
    print("Number of Line Segment : ", len(lines))
    #print("Houghlines : ", lines)

    for line in lines:
        x1,y1,x2,y2 = line[0]

        angle = 0.0
        angle = np.arctan2(y2-y1, x2-x1)*180. / np.pi
        dist = dist_calc(line)

        #cv2.line(rgb_img, (x1,y1), (x2,y2), (0,0,255), 1)

        if x1==x2 or y1==y2 or (angle < 10 and angle > -10):
            continue
        elif dist > dist_threshold:
            tmp = np.array([[dist]])
            tmp_line = np.concatenate((line[0], tmp[0]))
            new_line = tmp_line

            real_lines.append(new_line)
            #cv2.line(rgb_img, (x1,y1), (x2,y2), (0,255,0), 2)
        else:
            continue

    print("Number of Filtered Line : ", len(real_lines))

    if len(real_lines) < 3:
        return None

    # set camera_vector
    camera_vec = np.array([[(img_size_x+img_size_y)/2, 0, img_size_x/2],[0, (img_size_x+img_size_y)/2,img_size_y/2],[0, 0, 1]])

    sorted_lines = sorted(real_lines, key=itemgetter(4), reverse = True)

    interpretation_plane = []

    for lines in sorted_lines:
        x1,y1,x2,y2,dist = lines

        point_1 = np.array([[x1],[y1],[1]])
        line_point_1 = np.linalg.inv(camera_vec)
        line_point_1 = np.matmul(line_point_1, point_1)

        point_2 = np.array([[x2],[y2],[1]])
        line_point_2 = np.linalg.inv(camera_vec)
        line_point_2 = np.matmul(line_point_2, point_2)

        #print("line_point_1 : {}     line_point_2 : {}".format(line_point_1, line_point_2))

        cross_two_line = np.cross(line_point_1.transpose(), line_point_2.transpose())

        #print("cross_two_line : {}".format(cross_two_line))

        interpretation_plane.append(cross_two_line/np.sqrt(cross_two_line[0][0]**2+cross_two_line[0][1]**2+cross_two_line[0][2]**2))

    # Feature Vector == interpretation_plane

    feature_vec = np.array(interpretation_plane)
    feature_vec = np.transpose(feature_vec, (0,2,1))
    feature_vec = feature_vec[:,:,0]

    for i in range(len(feature_vec)):
        if feature_vec[i][2] < 0:
            #print("if문 : ", feature_vec[i])
            feature_vec[i] = -feature_vec[i]

    # print("feature_vec : {}".format(feature_vec))
    # print("feature_vec.shape : {}".format(feature_vec.shape))

    #model = DBSCAN(eps=0.9)
    model = KMeans(n_clusters=3, random_state=5)
    predict = model.fit_predict(feature_vec)

    print(predict)

    predict = np.reshape(predict, (len(predict),1))
    #predict = np.reshape(predict, (21,1))
    # print("predict.shape : {}".format(predict.shape))
    #print("interpretation_plane.shape : {}".format(np.asarray(interpretation_plane).shape))

    interpretation_plane = np.array(interpretation_plane)
    interpretation_plane = np.squeeze(interpretation_plane, axis=1)

    # print("interpretation_plane.shape : {}".format(interpretation_plane.shape))

    feature_vec = np.hstack([feature_vec, predict])
    interpretation_plane = np.hstack([interpretation_plane, predict])


    # print(interpretation_plane)
    # print(feature_vec.shape)

    for idx,lines in enumerate(sorted_lines):
        x1,y1,x2,y2,dist = lines

        if feature_vec[idx][3] == 0:
            cv2.line(rgb_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        elif feature_vec[idx][3] == 1:
            cv2.line(rgb_img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)
        elif feature_vec[idx][3] == 2:
            cv2.line(rgb_img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
        elif feature_vec[idx][3] == 3:
            cv2.line(rgb_img, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,255), 2)
        else: pass


    VD3D = []
    VD3D_label = []

    tmp_count = -1

    for i in range(0,len(interpretation_plane)):
        for j in range(i+1, len(interpretation_plane)):
            tmp_count += 1
            tmp_product = np.cross(interpretation_plane[i][:3],interpretation_plane[j][:3])

            tmp_product_norm = tmp_product/np.sqrt(tmp_product[0]**2+tmp_product[1]**2+tmp_product[2]**2)

            if tmp_product_norm[2] < 0.0:
                tmp_product_norm = -tmp_product_norm

            label_arr = np.array([tmp_count, i, j, interpretation_plane[i][3], interpretation_plane[j][3]])

            #print("label_arr : {}".format(label_arr))
            #print("tmp_product_norm : {}".format(tmp_product_norm))

            VD3D_label.append(label_arr)
            VD3D.append(tmp_product_norm)

    print(" * len(VD3D) : {}".format(len(VD3D)))

    # 2D Vanishing Point Candidates
    VP = []

    for i in range(len(VD3D)):
        tmp_vp = np.matmul(camera_vec, VD3D[i].transpose())
        if VD3D[i][2] != 0.:
            tmp_vp = tmp_vp/VD3D[i][2]
        else:
            tmp_vp = tmp_vp/0.000001

        if (tmp_vp[0] >= 0 and tmp_vp[0] <= img_size_x) or (tmp_vp[1] >= img_size_y and tmp_vp[1] <= img_size_y):
            cv2.circle(rgb_img, (int(tmp_vp[0]),int(tmp_vp[1])),3,(0,0,255),-1)
            #print("tmp_vp : {}".format(tmp_vp))
        VP.append(tmp_vp)




    score_matrix = np.zeros((len(VD3D),len(VD3D)))

    for i in range(len(VD3D)):
        for j in range(i+1, len(VD3D)):
            dist = np.linalg.norm(VP[i][:2]-VP[j][:2],2)

            if dist < 1:
                dist = 1

            score_matrix[i][j] = 1/dist
            score_matrix[j][i] = score_matrix[i][j]

    sum_score_matrix = score_matrix.sum(axis=1)
    sum_score_vector = sum_score_matrix.sum()

    score_vector = sum_score_matrix/sum_score_vector

    sorted_score_vector = np.sort(score_vector)[::-1]

    score_idx = []

    # if len(sorted_score_vector) == 0:
    #     return None

    for i in range(len(sorted_score_vector)):
        tmp_index = np.where(score_vector == sorted_score_vector[i])
        #print("tmp_index : {}".format(tmp_index))
        score_idx.append(tmp_index[0][0])

    if len(VP)//4 <= 0:
        return None

    numPointRank = len(VP)//4



    print("numPointRank : {}".format(numPointRank))

    if len(VP) < numPointRank:
        numPointRank = len(VP)

    topN_VP = []
    topN_label = []

    for i in range(numPointRank):
        #print("score_idx[i] : {}".format(score_idx[i]))
        topN_VP.append(VP[score_idx[i]][0:2])
        topN_label.append(VD3D_label[score_idx[i]])
        #cv2.circle(rgb_img, (int(VP[score_idx[i]][0]),int(VP[score_idx[i]][1])),3,(255,0,0),-1)

    print("len(topN_label) : {}".format(len(topN_label)))

    for label in topN_label:
        tmp_line_1 = sorted_lines[int(label[1])]
        tmp_label_1 = predict[int(label[1])]
        line1_x1, line1_y1, line1_x2, line1_y2,_ = tmp_line_1


        tmp_line_2 = sorted_lines[int(label[2])]
        tmp_label_2 = predict[int(label[2])]
        line2_x1, line2_y1, line2_x2, line2_y2,_ = tmp_line_2

        if tmp_label_1 == 0:
            cv2.line(van_img, (int(line1_x1),int(line1_y1)), (int(line1_x2),int(line1_y2)), (0,255,0), 2)
        elif tmp_label_1 == 1:
            cv2.line(van_img, (int(line1_x1),int(line1_y1)), (int(line1_x2),int(line1_y2)), (0,0,255), 2)
        elif tmp_label_1 == 2:
            cv2.line(van_img, (int(line1_x1),int(line1_y1)), (int(line1_x2),int(line1_y2)), (255,0,0), 2)
        elif tmp_label_1 == 3:
            cv2.line(van_img, (int(line1_x1),int(line1_y1)), (int(line1_x2),int(line1_y2)), (255,0,255), 2)
        else: pass

        if tmp_label_2 == 0:
            cv2.line(van_img, (int(line2_x1),int(line2_y1)), (int(line2_x2),int(line2_y2)), (0,255,0), 2)
        elif tmp_label_2 == 1:
            cv2.line(van_img, (int(line2_x1),int(line2_y1)), (int(line2_x2),int(line2_y2)), (0,0,255), 2)
        elif tmp_label_2 == 2:
            cv2.line(van_img, (int(line2_x1),int(line2_y1)), (int(line2_x2),int(line2_y2)), (255,0,0), 2)
        elif tmp_label_2 == 3:
            cv2.line(van_img, (int(line2_x1),int(line2_y1)), (int(line2_x2),int(line2_y2)), (255,0,255), 2)
        else: pass


    mean_topN_VP_X = np.mean(topN_VP[:], axis = 0)
    #mean_topN_VP_Y = np.mean(topN_VP[:])
    print(mean_topN_VP_X)

    # Vanishing Point Visualization
    cv2.circle(rgb_img, (int(mean_topN_VP_X[0]),int(mean_topN_VP_X[1])),4,(0,255,0),-1)
    cv2.circle(van_img, (int(mean_topN_VP_X[0]),int(mean_topN_VP_X[1])),4,(255,255,0),-1)

    #print(VD3D)
    #plot_circle(VD3D)
    #plot_3D_line(interpretation_plane)
    cv2.imshow("vanishing point attribute line segment",van_img)
    #cv2.waitKey(-1)

    return van_img

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    #cudnn.benchmark = True

    opt = opts.parse()
    warper = Warper()
    slidewindow = SlideWindow()

    print(("device id: {}".format(torch.cuda.current_device())))
    print("torch.version",torch.__version__)
    print("cuda_version",torch.version.cuda)


    models = importlib.import_module('models.init')
    # print(models)
    criterions = importlib.import_module('criterions.init')
    checkpoints = importlib.import_module('checkpoints')
    Trainer = importlib.import_module('models.' + opt.netType + '-train')

    # Data loading
    print('=> Setting up data loader')

    # Load previous checkpoint, if it exists
    print('=> Checking checkpoints')
    checkpoint = checkpoints.load(opt)

    # Create model
    model, optimState = models.setup(opt, checkpoint)
    model.cuda()

    criterion = criterions.setup(opt, checkpoint, model)

    ##################################################################################

    model.eval()

    cap = cv2.VideoCapture("input_video/Driving_Studio.avi")

    if cap.isOpened():
        print("width : {}, height : {}".format(cap.get(3), cap.get(4)))

    video_width = int(cap.get(3))
    video_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('output_video/vanishing_Driving_Studio.avi', fourcc, 20.0, (640,480))

    prev_time = 0

    fps_list = []

    while True:
        ret, frame = cap.read()

        if ret:
            cur_time = time.time()
            frame_new = cv2.resize(frame, (640,480))

            input_img = frame_new / 255.
            input_img = preprocess_img(input_img)

            # array to tensor
            input_img = torch.from_numpy(input_img).float()

            with torch.no_grad():
                inputData_var = Variable(input_img).unsqueeze(0).cuda()

                # inference
                output = model.forward(inputData_var)
                output = torch.sigmoid(output)
                #output = F.softmax(output, dim=1)




                # gpu -> cpu,  tensor -> numpy
                output = output.detach().cpu().numpy()


                #print(np.unique(output))

                output = output[0]

                output = postprocess_img(output)
                output = np.clip(output, 0, 1)
                output *= 255
                output = np.uint8(output)


                output = cv2.resize(output, (640, 480))
                # output[output>=120] = 255
                # output[output<120] = 0
                # output[output<=40] = 0

                #line_img = vanishing_point_detector(frame, output)



                #ret, thr_img = cv2.threshold(output, 50, 255, cv2.THRESH_BINARY)
                warper_img = warper.warp(output)
                ret, left_start_x, right_start_x, cf_img = slidewindow.w_slidewindow(warper_img)

                # if line_img is not None:
                #     out.write(line_img)



                end_time = time.time()
                sec = end_time - cur_time


                # add_img = cv2.hconcat([frame, output])
                # print("add_img.shape : ", add_img.shape)

                fps = 1/sec
                fps_list.append(fps)

                print("Estimated fps {0} " . format(fps))

                # out.write(add_img)

                cv2.imshow("frame",frame)
                # cv2.imshow("src", warper_img)
                cv2.imshow("out_img", output)
                cv2.imshow("cf_img", cf_img)
                #cv2.imshow("canny_img", erode_img)
                # cv2.imshow("out_img2", otsu_img_2)

                key = cv2.waitKey(1) & 0xFF
                if key == 27: break
                elif key == ord('p'):
                    cv2.waitKey(-1)



if __name__ == '__main__':
    main()
