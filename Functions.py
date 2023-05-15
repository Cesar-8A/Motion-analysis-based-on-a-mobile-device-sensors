def mediapipe_estimation(path):
    import cv2
    import mediapipe as mp
    import numpy as np
  
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    data = {
        "NOSE":[],
        "LEFT_EYE_INNER":[],
        "LEFT_EYE":[],
        "LEFT_EYE_OUTER":[],
        "RIGHT_EYE_INNER":[],
        "RIGHT_EYE":[],
        "RIGHT_EYE_OUTER":[],
        "LEFT_EAR":[],
        "RIGHT_EAR":[],
        "MOUTH_LEFT":[],
        "MOUTH_RIGHT":[],
        "LEFT_SHOULDER":[],
        "RIGHT_SHOULDER":[],
        "LEFT_ELBOW":[],
        "RIGHT_ELBOW":[],
        "LEFT_WRIST":[],
        "RIGHT_WRIST":[],
        "LEFT_PINKY":[],
        "RIGHT_PINKY":[],
        "LEFT_INDEX":[],
        "RIGHT_INDEX":[],
        "LEFT_THUMB":[],
        "RIGHT_THUMB":[],
        "LEFT_HIP":[],
        "RIGHT_HIP":[],
        "LEFT_KNEE":[],
        "RIGHT_KNEE":[],
        "LEFT_ANKLE":[],
        "RIGHT_ANKLE":[],
        "LEFT_HEEL":[],
        "RIGHT_HEEL":[],
        "LEFT_FOOT_INDEX":[],
        "RIGHT_FOOT_INDEX":[]
    }

    #For video
    cap = cv2.VideoCapture(path)
    # Obtiene la velocidad de fotogramas (fps)

    x = 0
    y = 1

    if not cap.isOpened():
        print('Failed opening video')
    else:
        print("Starting...")
        empty = []
        images = []
        frame_num = 0
        with mp_pose.Pose(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.75,
                smooth_landmarks=True, 
                model_complexity = 2) as pose:
            while cap.isOpened():
                success, image = cap.read()
                frame_num += 1
                if not success:
                    print("End of video.")
                    break
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)
                #Store the data
                if  isinstance(results.pose_landmarks, type(None)):
                #Adding frame with its index
                    empty.append(frame_num)
                else:
                    for key, i in zip(data.keys(), range(33)):
                        data[key].append([results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z])
                #Draw the landmarks
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)      
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                #Save the image in a mtrix
                images.append(image)
            # Flip the image horizontally for a selfie-view display.
            #Make them numpy
            for key in data.keys():
                data[key] = np.array(data[key])
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

    for keys in data.keys():
        for i in range(len(data[keys][:])):
            cv2.circle(images[i], (int(data[keys][i,x]*width), int(data[keys][i,y]*height)), radius=5, color=[255,0,0], thickness=-1)
    return(images,width,height,data,keys)


def angle_estimation(vertex1, vertex2, vertex3):
    import math
    # Calcula la longitud de cada lado del triángulo
    a = math.dist(vertex3, vertex2)
    b = math.dist(vertex1, vertex3)
    c = math.dist(vertex2, vertex1)
    # Calcula el coseno del ángulo central correspondiente al lado b
    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
    # Calcula el ángulo central correspondiente al lado b
    angle = math.acos(cos_B)
    angle = math.degrees(angle)
    angle = abs(angle)
    return(angle)

def video_exportation(images,name):
    import moviepy.editor as mpy
    # Crear un clip de ejemplo con un par de cuadros
    fps = 30  # cuadros por segundo
    # Crear un clip a partir de los cuadros
    clip = mpy.ImageSequenceClip(images, fps=fps)
     # Guardar el clip en formato mp4
    clip.write_videofile(name)