from flask import Flask, request, jsonify
import math
from sklearn import neighbors
import os
import os.path
import pickle
import configparser
import inspect
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import sys, json
import cv2
import pandas as pd
from datetime import datetime
from PIL import Image
from io import BytesIO
import platform
import numpy as np

# Init Flask app
app = Flask(__name__)


def bytes_to_img(bytes_array):
    """
    Function to conver bytes array to img type

    :param bytes_array:  array of bytes from image
    :return: Image object

    """
    stream = BytesIO(bytes_array)
    image = Image.open(stream).convert("RGBA")

    return image


def path_creator(rel_path=''):
    """
    Creates an absolute path to object

    :param rel_path: (optional) relative path to object
    :return: absolute path to object

    """
    if platform.system() != 'Windows':
        if rel_path == '':
            path_list=sys.argv[0].split('/')[:-1]
            return '/'.join(path_list)
        else:
            path_list = sys.argv[0].split('/')[:-1]
            return '/'.join(path_list) + '/' + rel_path
    else:
        if rel_path == '':
            path_list=sys.argv[0].split('\\')[:-1]
            path_res='\\'.join(path_list)
            return path_res
        else:
            path_list = sys.argv[0].split('\\')[:-1]
            rel_path=rel_path.split('/')
            path_res='\\'.join(path_list) + '\\' + '\\'.join(rel_path)
            return path_res



def train_knn_model(train_dir,
                    model_save_path=None,
                    n_neighbors=None,
                    knn_algo='ball_tree',
                    verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.
    (View in source code to see train_dir example tree structure)
    Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """

    x = []
    y = []

    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the training set
                x.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)


    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(x))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    print("Training KNN classifier...")
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(x, y)

    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            pickle.dump(knn_clf, f)

    print("Training complete!")
    return knn_clf


def config_gettype(function_name,
                   config_name,
                   param):
    """
    Function for getting configs from .INI file with right types"

    :param function_name: main key for INI file to extract right func params
    :param config_name: name (relative path) of config
    :param param: param (key value of INI) to extract
    :return: value of param with right type

    """
    config = configparser.ConfigParser()
    config.read(path_creator(config_name))
    #config.read(path_creator(config_name))
    if config[function_name][param].split(' ## ')[1] == 'str':
        return str(config.get(function_name,param).split(' ## ')[0])
    if config[function_name][param].split(' ## ')[1] == 'int':
        return int(config.get(function_name,param).split(' ## ')[0])
    if config[function_name][param].split(' ## ')[1] == 'float':
        return float(config.get(function_name,param).split(' ## ')[0])
    if config[function_name][param].split(' ## ')[1] == 'bool':
        return bool(config.get(function_name,param).split(' ## ')[0])
    if config[function_name][param].split(' ## ')[1] == 'path':
        return path_creator(str(config.get(function_name,param).split(' ## ')[0]))
    if config[function_name][param].split(' ## ')[1] == 'NoneType':
        return None


def recognize_faces(x_img,
                    knn_clf=None,
                    model_path=None,
                    distance_threshold=0.3):
    """
    Recognizes faces in given image using a trained KNN classifier

    :param x_img: image to work with
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.

    """
    print("Start recognize")
    # Making a check
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thought knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    # Load image file and find face locations
    x_face_locations = face_recognition.face_locations(x_img)
        # Set variable for changes on camera (if connected) check
        # x_face_locations_len = 0

    # If no faces are found in the image, return an empty result
    if len(x_face_locations) == 0:
        return []
      

    # Checking for changes on camera (if connected)
    # if len(x_face_locations) != x_face_locations_len:
       # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(x_img, known_face_locations=x_face_locations)
    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(x_face_locations))]
    accur_list = [1-closest_distances[0][i][0] for i in range(len(x_face_locations))]
    x_face_locations_len = len(x_face_locations)
    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc, accur, rec) if rec else ("unknown", loc, 0,0) for pred, loc, accur, rec in
            zip(knn_clf.predict(faces_encodings),
                x_face_locations,
                accur_list,
                are_matches)]
    


def read_txt(path):
    """
    Function for finding and reading text information of persons from faces_train_data
    :param path: path to faces
    :return: dictionary of all founded and read txt files with name as key

    """
    path=path_creator(path)
    names_dic = {}
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".txt"):
                names_dic.update([(root.split('/')[2], os.path.join(root, file))])
    return names_dic


def faces_info_export(frame):
    """
    Gets an image, recognize faces and returns df with information about it

    :param frame: formatted frame with faces to recognize
    :return: time code of recognition, name, accuracy, image path (recource)

    """
    faces_info_dict = {}
    faces_info_dict.setdefault('name', [])
    # faces_info_dict.setdefault('Info', [])
    faces_info_dict.setdefault('time_mark', [])
    # faces_info_dict.setdefault('image_info', [])
    faces_info_dict.setdefault('accuracy', [])
    faces_info_dict.setdefault('face_on_cam', [])
    # faces_info_dict.setdefault('employee_info', [])

    # path_of_img = frame
    #print(frame)
    # frame = cv2.imread(path_of_img)
    #cv2.imshow('parh', frame)
    #print(frame)
    # Для более быстрой обработки измениним размер в 1/4 раза
    #small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    # Конвертируем цветвоую схему получаемого изображения
    #rgb_small_frame = small_frame[:, :, ::-1]
    try:
        rgb_frame = frame[:,:,::-1].copy()
        recognize_faces_params = [config_gettype('recognize_faces', 'FRS.ini', param) for param in
                              inspect.getfullargspec(recognize_faces)[0]]
        recognize_faces_params.remove('rgb_small_frame')
        recognize_faces_params.insert(0, rgb_frame)
        predictions = recognize_faces(*recognize_faces_params)

    # txt_path = [config_gettype('read_txt', 'FRS.ini', 'path')]
    # txt_path=str(txt_path)
        for name, _, accur, rec in predictions:
            faces_info_dict['name'].append(name)
            faces_info_dict['time_mark'].append(datetime.now())
        # faces_info_dict['image_info'].append(str(path_of_img))
            faces_info_dict['accuracy'].append(float(accur))
            faces_info_dict['face_on_cam'].append(bool(rec))

        # if name != 'unknown':
        #     faces_info_dict['employee_info'].append(str(open(read_txt(txt_path)[name]).read()))
        # else:
        #     faces_info_dict['employee_info'].append('no_info')

        faces_info_df = pd.DataFrame.from_dict(faces_info_dict)
        faces_info_df.to_csv('faces_info_csv')
        return faces_info_df
    except TypeError as e:
        print('None')
    

# API functions below _____________________________________
@app.route('/FRS/FRS_recognition', methods=['POST'])
def main_recognition():
    """
    Main function for recognition API

    :return: df in json format

    """
    if request.method == 'POST':
        # print(request.url)
        # stream = BytesIO(request.data)
        # image = Image.open(stream).convert("RGBA")
        # path = 'C:/Users/13/Documents/FRS_v1/path.png'
        # image = image.save(path)
        # stream.close()
        #df = faces_info_export(path)
        print(request.url)
        stream = BytesIO(request.data)
        img_pil=Image.open(stream).convert("RGB")
        stream.close()
        img_cv=np.array(img_pil)
        try:
            df = faces_info_export(img_cv)
            return df.to_json(orient='index')
        except SystemError as er:
        	print(er)
        	return json.dumps({'msg':'error'})
        except AttributeError as er:
        	print(er)
        	return json.dumps({'msg':'error'})
    if request.method == 'GET':
        # ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
        df = faces_info_export("C:/Users/13/Documents/FRS_v1/test_image.jpg")
        return df.to_json(orient='index')


@app.route('/FRS/FRS_training', methods=['GET'])
def main_training():
    """
    Main function for training API
    :return: message of successful model training
    """
    if request.method == 'GET':
        print("Working directory: ", path_creator())
        train_knn_model_params=[config_gettype('train_knn_model','FRS.ini',param) for param in inspect.getfullargspec(train_knn_model)[0]]
        train_knn_model(*train_knn_model_params)
        return_text="FRS_training_model.py completed"
        return jsonify(return_text)
    else:
        return_text1 = "Опа"
        return jsonify(return_text1)



if __name__ == "__main__":
    app.run(host='0.0.0.0', port='5001', debug=True)
