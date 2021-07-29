import os
import xml.etree.ElementTree as ET
from os import getcwd


def first_step():
    save_base_path = r'./ImageSet/Main'

    train_xml_file_path = r'./ImageSet/Annotations/train'
    train_temp_xml = os.listdir(train_xml_file_path)
    train_total_xml = []

    val_xml_file_path = r'./ImageSet/Annotations/val'
    val_temp_xml = os.listdir(val_xml_file_path)
    val_total_xml = []

    test_xml_file_path = r'./ImageSet/Annotations/test'
    test_temp_xml = os.listdir(test_xml_file_path)
    test_total_xml = []

    for xml in train_temp_xml:
        if xml.endswith(".xml"):
            train_total_xml.append(xml)

    for xml in val_temp_xml:
        if xml.endswith(".xml"):
            val_total_xml.append(xml)

    for xml in test_temp_xml:
        if xml.endswith(".xml"):
            test_total_xml.append(xml)

    train_num = len(train_total_xml)
    train_ser_num = range(train_num)
    print("train size", train_num)
    val_num = len(val_total_xml)
    val_ser_num = range(val_num)
    print("val size", val_num)
    test_num = len(test_total_xml)
    test_ser_num = range(test_num)
    print("test size", test_num)
    f_test = open(os.path.join(save_base_path, 'test.txt'), 'w')
    f_train = open(os.path.join(save_base_path, 'train.txt'), 'w')
    f_val = open(os.path.join(save_base_path, 'val.txt'), 'w')

    for i in train_ser_num:
        name = train_total_xml[i][:-4]+'\n'
        f_train.write(name)
    for i in val_ser_num:
        name = val_total_xml[i][:-4]+'\n'
        f_val.write(name)
    for i in test_ser_num:
        name = test_total_xml[i][:-4]+'\n'
        f_test.write(name)

    f_test.close()
    f_train.close()
    f_val.close()


sets = ['train', 'val', 'test']
classes = ['S', 'Impurity']


def convert_annotation(img_id, l_file, image_set):
    in_file = open('ImageSet/Annotations/%s/%s.xml' %(image_set, img_id), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xml_box = obj.find('bndbox')
        b = (int(xml_box.find('xmin').text), int(xml_box.find('ymin').text), int(xml_box.find('xmax').text),
             int(xml_box.find('ymax').text))
        l_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


def second_step():
    wd = getcwd()
    for image_set in sets:
        image_ids = open('./ImageSet/Main/%s.txt' % image_set).read().strip().split()
        list_file = open('%s.txt' % image_set, 'w')
        for image_id in image_ids:
            list_file.write('%s/ImageSet/PNGImages/Img/%s.png' % (wd, image_id))
            convert_annotation(image_id, list_file, image_set)
            list_file.write('\n')
        list_file.close()
    print('completed done!!!')
