def squat_ana(path): 
  import cv2
  import mediapipe as mp
  import numpy as np
  import moviepy.editor as mpy
  import math
  import matplotlib.pyplot as plt

  def central_angle(vertex1, vertex2, vertex3):
      # Calcula la longitud de cada lado del triángulo
      a = math.dist(vertex1, vertex2)
      b = math.dist(vertex2, vertex3)
      c = math.dist(vertex3, vertex1)
      # Calcula el coseno del ángulo central correspondiente al lado b
      cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
      # Calcula el ángulo central correspondiente al lado b
      angle = 2 * math.acos(cos_B)
      angle = math.degrees(angle)
      if(angle >= 90):
        angle = abs((angle - 180))
      # Convierte el ángulo a grados y lo devuelve
      return(angle)

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
  frames_amount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 

  x = 0
  y = 1
  deep_hip = []
  knee_wrong = []
  step = 0
  step_brake_knee = 0
  step_brake_hip = 0
  step_brake = 0
  holder = 0

  fps = cap.get(cv2.CAP_PROP_FPS)
  if not cap.isOpened():
    print('Failed opening video')
  else:
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
          # Adding frame with its index
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
          if(((results.pose_landmarks.landmark[25].y - results.pose_landmarks.landmark[23].y) < -0.030) or ((results.pose_landmarks.landmark[26].y - results.pose_landmarks.landmark[24].y) < -0.030 )):
            deep_hip.append(step)
            step_brake_hip = 1
            cv2.putText(image, "Too deep hip movement" , (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
          A = np.array([results.pose_landmarks.landmark[31].x, results.pose_landmarks.landmark[31].y])
          C = np.array([results.pose_landmarks.landmark[25].x, results.pose_landmarks.landmark[25].y])
          B = np.array([results.pose_landmarks.landmark[27].x, results.pose_landmarks.landmark[27].y])
          angle_calculated_left = central_angle(A, B, C)
          
          A = np.array([results.pose_landmarks.landmark[32].x, results.pose_landmarks.landmark[32].y])
          C = np.array([results.pose_landmarks.landmark[26].x, results.pose_landmarks.landmark[26].y])
          B = np.array([results.pose_landmarks.landmark[28].x, results.pose_landmarks.landmark[28].y])
          angle_calculated_right = central_angle(A, B, C)
          angle_calculated = (angle_calculated_left + angle_calculated_right) / 2
          if (angle_calculated < 55):
            knee_wrong.append(step)
            step_brake_knee = 1
            cv2.putText(image, "Wrong knee position" , (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
        step += 1
        #cv2.imshow('Process',image)
        plt.imshow(image)
        plt.pause(0.01)

        if(step_brake_hip or step_brake_knee):
          step_brake = 1
        else:
          step_brake = 0
          holder = 0

        if(step_brake == 1 and holder == 0):
          holder = 1
          print("Error detected")
          input()
        step_brake_hip = 0
        step_brake_knee = 0
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

  #Print in terminal movement evaluation
  knee_errors_percent = 100 - (len(knee_wrong) / frames_amount) *100
  hip_errors_percent = 100 - (len(deep_hip) / frames_amount) * 100
  print("Se tuvo una presición de la rodilla en un %.2f " % (knee_errors_percent))
  print("Se tuvo una presición de la cadera en un %.2f " % (hip_errors_percent))

  # Crear un clip de ejemplo con un par de cuadros
  W, H = 1280, 720
  duration = 6  # duración del video en segundos
  fps = 30  # cuadros por segundo
  num_frames = duration * fps

  # Crear un clip a partir de los cuadros
  clip = mpy.ImageSequenceClip(images, fps=fps)
  # Guardar el clip en formato mp4
  clip.write_videofile('squat_2_procesado.mp4')

squat_ana("D:\squat_2.mp4")
