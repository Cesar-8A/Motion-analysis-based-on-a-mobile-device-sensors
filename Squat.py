def squat_ana(path,Video_label):
  import cv2
  import mediapipe as mp
  import numpy as np
  import moviepy.editor as mpy
  import math
  import tkinter
  from PIL import Image
  from PIL import ImageTk
  import time
  import matplotlib.pyplot as plt
  from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

  fps = cap.get(cv2.CAP_PROP_FPS)
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

  #Too deep down hips movement
  deep_hip = []
  for i in range(frames_amount - 1):
    if(((data["LEFT_KNEE"][i,y] - data["LEFT_HIP"][i,y]) < -0.030) or ((data["RIGHT_KNEE"][i,y] - data["RIGHT_HIP"][i,y]) < -0.030)):
      deep_hip.append(i)

  #Wrong knee movement
  knee_wrong = []
  for i in range(frames_amount - 1):
    A = np.array([data["LEFT_FOOT_INDEX"][i,x], data["LEFT_FOOT_INDEX"][i,y]])
    C = np.array([data["LEFT_KNEE"][i,x], data["LEFT_KNEE"][i,y]])
    B = np.array([data["LEFT_ANKLE"][i,x], data["LEFT_ANKLE"][i,y]])
    angle_calculated_left = central_angle(A, B, C)
    
    A = np.array([data["RIGHT_FOOT_INDEX"][i,x], data["RIGHT_FOOT_INDEX"][i,y]])
    C = np.array([data["RIGHT_KNEE"][i,x], data["RIGHT_KNEE"][i,y]])
    B = np.array([data["RIGHT_ANKLE"][i,x], data["RIGHT_ANKLE"][i,y]])
    angle_calculated_right = central_angle(A, B, C)
    angle_calculated = (angle_calculated_left + angle_calculated_right) / 2
    if (angle_calculated < 55):
      knee_wrong.append(i)

  #Print in terminal movement evaluation
  knee_errors_percent = 100 - (len(knee_wrong) / frames_amount) *100
  hip_errors_percent = 100 - (len(deep_hip) / frames_amount) * 100
  print("Se tuvo una presición de la rodilla en un %.2f " % (knee_errors_percent))
  print("Se tuvo una presición de la cadera en un %.2f " % (hip_errors_percent))

  #Video anotations
  for i in range(len(knee_wrong)):
    editando = images[knee_wrong[i]]
    cv2.putText(editando, "Wrong knee position" , (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
    images[knee_wrong[i]] = editando
  for i in range(len(deep_hip)):
    editando = images[knee_wrong[i]]
    cv2.putText(editando, "Too deep hip movement" , (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
    images[knee_wrong[i]] = editando

  # Crear un clip de ejemplo con un par de cuadros
  W, H = 1280, 720
  duration = 6  # duración del video en segundos
  fps = 30  # cuadros por segundo
  num_frames = duration * fps

  # Crear un clip a partir de los cuadros
  clip = mpy.ImageSequenceClip(images, fps=fps)
  # Guardar el clip en formato mp4
  clip.write_videofile('squat_procesado.mp4')

  #show in tkinter window
  global step_brake
  step_brake = 0
  global p
  p = 0
  def show():
    global step_brake
    global p
    if(p < len(images)):
      im_pil = Image.fromarray(np.uint8(images[p]))
      im_pil = im_pil.resize((int(width / 2), int(height / 2)),Image.Resampling.NEAREST)
      img = ImageTk.PhotoImage(image=im_pil)
      Video_label.configure(image=img)
      Video_label.image = img

      if(((p in knee_wrong) or (p in deep_hip)) and step_brake == 0):
        step_brake = 1
        print("Error detected")
        input()
      if(not (p in knee_wrong or p in deep_hip) and step_brake == 1):
        step_brake = 0
      p += 1
      Video_label.after(40, show)
  show()