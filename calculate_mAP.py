import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


def Get_map():
    MINOVERLAP = 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
    parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
    parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
    parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
    parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
    args = parser.parse_args()

    if args.ignore is None:
        args.ignore = []

    specific_iou_flagged = False
    if args.set_class_iou is not None:
        specific_iou_flagged = True

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth')
    DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results')
    IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')
    if os.path.exists(IMG_PATH):
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    show_animation = False
    if not args.no_animation:
        try:
            show_animation = True
        except ImportError:
            print("\"opencv-python\" not found, please install to visualize the results.")
            args.no_animation = True

    draw_plot = False
    if not args.no_plot:
        try:
            draw_plot = True
        except ImportError:
            print("\"matplotlib\" not found, please install it to get the resulting plots.")
            args.no_plot = True

    def log_average_miss_rate(precision, fp_cumsum, num_images):
        if precision.size == 0:
            lam = 0
            mr1 = 1
            fpi = 0
            return lam, mr1, fpi
        fpi = fp_cumsum / float(num_images)
        mr1 = (1 - precision)
        fppi_tmp = np.insert(fpi, 0, -1.0)
        mr_tmp = np.insert(mr1, 0, 1.0)
        ref = np.logspace(-2.0, 0.0, num=9)
        for i, ref_i in enumerate(ref):
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]
        lam = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lam, mr1, fpi

    def error(msg):
        print(msg)
        sys.exit(0)

    def is_float_between_0_and_1(value):
        try:
            val1 = float(value)
            if 0.0 < val1 < 1.0:
                return True
            else:
                return False
        except ValueError:
            return False

    def voc_ap(Rec, Prec):
        Rec.insert(0, 0.0)
        Rec.append(1.0)
        mrec = Rec[:]
        Prec.insert(0, 0.0)
        Prec.append(0.0)
        mpre = Prec[:]

        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i - 1]:
                i_list.append(i)

        ap = 0.0
        for i in i_list:
            ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
        return ap, mrec, mpre

    def file_lines_to_list(path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content

    def draw_text_in_image(img, text, pos, color, line_width):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 1
        bottomLeftCornerOfText = pos
        cv2.putText(img, text,
                    bottomLeftCornerOfText,
                    font,
                    fontScale,
                    color,
                    lineType)
        text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
        return img, (line_width + text_width)

    def adjust_axes(r, t, fig, axes):
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1] * propotion])

    def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                       true_p_bar):
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        if true_p_bar != "":
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])
            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive',
                     left=fp_sorted)
            plt.legend(loc='lower right')
            fig = plt.gcf()
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values) - 1):  # largest bar
                    adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            fig = plt.gcf()
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val)
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                if i == (len(sorted_values) - 1):
                    adjust_axes(r, t, fig, axes)
        fig.canvas.set_window_title(window_title)
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        init_height = fig.get_figheight()
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4)
        height_in = height_pt / dpi
        top_margin = 0.15  # in percentage of the figure height
        bottom_margin = 0.05  # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        plt.title(plot_title, fontsize=14)
        plt.xlabel(x_label, fontsize='large')
        fig.tight_layout()
        fig.savefig(output_path)
        if to_show:
            plt.show()
        plt.close()

    def draw_gtbox_in_image(GT_PATH, results_files_path):
        light_blue = (255, 200, 100)
        for gt_file in os.listdir(GT_PATH):
            if gt_file.endswith("truth.json"):
                gt_file = GT_PATH + "/" + gt_file
                ground_truth_data = json.load(open(gt_file))
                font = cv2.FONT_HERSHEY_SIMPLEX
                img = cv2.imread("input/images-optional" + "/" + gt_file[-29:-18] + ".png")
                img_cumulative_path = results_files_path + "/images/" + gt_file[-29:-18] + ".png"
                if os.path.isfile(img_cumulative_path):
                    img_cumulative = cv2.imread(img_cumulative_path)
                else:
                    img_cumulative = img.copy()
                for obj in ground_truth_data:
                    class_name = obj["class_name"]
                    bbgt = [int(round(float(x))) for x in obj["bbox"].split()]
                    cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 1)
                    cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                cv2.LINE_AA)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
    results_files_path = "results"
    if os.path.exists(results_files_path):
        shutil.rmtree(results_files_path)

    os.makedirs(results_files_path)
    if draw_plot:
        os.makedirs(os.path.join(results_files_path, "classes"))
    if show_animation:
        os.makedirs(os.path.join(results_files_path, "images", "detections_one_by_one"))

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name, left, top, right, bottom, _difficult = line.split()
                    is_difficult = True
                else:
                    class_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False, "difficult": True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name": class_name, "bbox": bbox, "used": False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        with open(TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    if specific_iou_flagged:
        n_args = len(args.set_class_iou)
        error_msg = '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
        if n_args % 2 != 0:
            error('Error, missing arguments. Flag usage:' + error_msg)
        specific_iou_classes = args.set_class_iou[::2]
        iou_list = args.set_class_iou[1::2]
        if len(specific_iou_classes) != len(iou_list):
            error('Error, missing arguments. Flag usage:' + error_msg)
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            file_id = txt_file.split(".txt", 1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH, (file_id + ".txt"))
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence": confidence, "file_id": file_id, "bbox": bbox})
        bounding_boxes.sort(key=lambda x: float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    draw_gtbox_in_image(TEMP_FILES_PATH, results_files_path)
    sum_AP = 0.0
    Accuracy = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    with open(results_files_path + "/results.txt", 'w') as results_file:
        results_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            for idx, detection in enumerate(dr_data):
                file_id = detection["file_id"]
                if show_animation:
                    ground_truth_img = glob.glob1(IMG_PATH, file_id + ".*")
                    if len(ground_truth_img) == 0:
                        error("Error. Image not found with id: " + file_id)
                    elif len(ground_truth_img) > 1:
                        error("Error. Multiple image with id: " + file_id)
                    else:
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                        img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        bottom_border = 60
                        BLACK = [0, 0, 0]
                        img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                distance = 1e8
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj["bbox"].split()]
                        bi = [max(bb[0], bbgt[0]), max(bb[1], bbgt[1]), min(bb[2], bbgt[2]), min(bb[3], bbgt[3])]

                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        center_gt_x = (bbgt[0] + bbgt[2]) / 2
                        center_gt_y = (bbgt[1] + bbgt[3]) / 2
                        center_dr_x = (bb[0] + bb[2]) / 2
                        center_dr_y = (bb[1] + bb[3]) / 2
                        D = ((center_dr_x - center_gt_x)**2 + (center_dr_y - center_gt_y)**2)**0.5
                        if D < distance:
                            distance = D
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                                              + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih / ua
                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj

                if show_animation:
                    status = "NO MATCH FOUND!"
                # set minimum overlap
                min_overlap = MINOVERLAP
                if specific_iou_flagged:
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        min_overlap = float(iou_list[index])
                if ovmax >= min_overlap or (ovmax >= min_overlap-0.1 and distance <= 3):
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file, 'w') as f:
                                f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"
                else:
                    fp[idx] = 1
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"

                if show_animation:
                    height, widht = img.shape[:2]
                    white = (255, 255, 255)
                    light_blue = (255, 200, 100)
                    green = (0, 255, 0)
                    light_red = (30, 30, 255)
                    margin = 10
                    v_pos = int(height - margin - (bottom_border / 2.0))
                    text = "Image: " + ground_truth_img[0] + " "
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
                    if ovmax != -1:
                        color = light_red
                        if status == "INSUFFICIENT OVERLAP":
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + "< {0:.2f}% ".format(min_overlap * 100)
                        else:
                            text = "IoU: {0:.2f}% ".format(ovmax * 100) + ">= {0:.2f}% ".format(min_overlap * 100)
                            color = green
                        img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
                    v_pos += int(bottom_border / 2.0)
                    rank_pos = str(idx + 1)
                    text = "Detection #rank: " + rank_pos + " confidence: {0:.2f}% ".format(
                        float(detection["confidence"]) * 100)
                    img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
                    color = light_red
                    if status == "MATCH!":
                        color = green
                    text = "Result: " + status + " "
                    img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)

                    font = cv2.FONT_HERSHEY_SIMPLEX
                    if ovmax > 0:
                        bbgt = [int(round(float(x))) for x in gt_match["bbox"].split()]
                        cv2.rectangle(img, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 1)
                        cv2.rectangle(img_cumulative, (bbgt[0], bbgt[1]), (bbgt[2], bbgt[3]), light_blue, 1)
                        cv2.putText(img_cumulative, class_name, (bbgt[0], bbgt[1] - 5), font, 0.6, light_blue, 1,
                                    cv2.LINE_AA)
                    bb = [int(i) for i in bb]
                    cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)
                    cv2.rectangle(img_cumulative, (bb[0], bb[1]), (bb[2], bb[3]), color, 1)
                    cv2.putText(img_cumulative, class_name, (bb[0], bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
                    cv2.imshow("Animation", img)
                    cv2.waitKey(1)
                    cv2.imwrite(img_cumulative_path, img_cumulative)

            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            text = "{0:.2f}%".format(ap * 100) + " = " + class_name + " AP "
            rounded_prec = ['%.2f' % elem for elem in prec]
            rounded_rec = ['%.2f' % elem for elem in rec]
            results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :"
                               + str(rounded_rec) + "\n\n")
            if not args.quiet:
                print(text)
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(rec), np.array(fp), n_images)

            lamr_dictionary[class_name] = lamr
            if draw_plot:
                plt.plot(rec, prec, '-o')
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                fig = plt.gcf()
                fig.canvas.set_window_title('AP ' + class_name)
                plt.title('class: ' + text)
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                axes = plt.gca()
                axes.set_xlim([0.0, 1.0])
                axes.set_ylim([0.0, 1.05])
                fig.savefig(results_files_path + "/classes/" + class_name + ".png")
                plt.cla()

        if show_animation:
            cv2.destroyAllWindows()

        results_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP = {0:.2f}%".format(mAP * 100)
        results_file.write(text + "\n")
        print(text)
    shutil.rmtree(TEMP_FILES_PATH)
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            if class_name in args.ignore:
                continue
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                det_counter_per_class[class_name] = 1
    dr_classes = list(det_counter_per_class.keys())
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
        x_label = "Number of objects per class"
        output_path = results_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of ground-truth objects per class\n")
        for class_name in sorted(gt_counter_per_class):
            results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")

    for class_name in dr_classes:
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    if draw_plot:
        window_title = "detection-results-info"
        plot_title = "detection-results\n"
        plot_title += "(" + str(len(dr_files_list)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(det_counter_per_class.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        x_label = "Number of objects per class"
        output_path = results_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
        )

    with open(results_files_path + "/results.txt", 'a') as results_file:
        results_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            results_file.write(text)

    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = results_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )

    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP * 100)
        x_label = "Average Precision"
        output_path = results_files_path + "/mAP.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
        )
