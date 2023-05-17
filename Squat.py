def squat_ana(path,Video_label,Info_posture,Load_button):
  import cv2
  import mediapipe as mp
  import numpy as np
  import moviepy.editor as mpy
  import math
  from PIL import Image
  from PIL import ImageTk
  import Functions as f
  
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
  Info_posture.grid_remove()
  Load_button.grid_remove()
  # Obtiene la velocidad de fotogramas (fps)

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

  #Detection of start pose
  distance_shoulder_heel = []

  for i in range(len(data[keys][:])):
    a = ((data["RIGHT_HEEL"][i,y] - data["RIGHT_SHOULDER"][i,y]) + (data["LEFT_HEEL"][i,y] - data["LEFT_SHOULDER"][i,y]) / 2)
    distance_shoulder_heel.append(a)

  holder = min(distance_shoulder_heel)

  for i in range(len(data[keys][:])):
    if (abs(holder - distance_shoulder_heel[i]) > 0.2):
      distance_shoulder_heel[i] = 0
  #Too deep down hips movement
  deep_hip = []
  for i in range(len(data[keys][:])):
    if(distance_shoulder_heel[i] > 0):
      A = np.array([data["LEFT_HIP"][i,x], data["LEFT_HIP"][i,y]])
      B = np.array([data["LEFT_KNEE"][i,x], data["LEFT_KNEE"][i,y]])
      C = np.array([data["LEFT_ANKLE"][i,x], data["LEFT_ANKLE"][i,y]])
      angle_calculated_left = f.angle_estimation(A, B, C)
      
      A = np.array([data["RIGHT_HIP"][i,x], data["RIGHT_HIP"][i,y]])
      B = np.array([data["RIGHT_KNEE"][i,x], data["RIGHT_KNEE"][i,y]])
      C = np.array([data["RIGHT_ANKLE"][i,x], data["RIGHT_ANKLE"][i,y]])
      angle_calculated_right = f.angle_estimation(A, B, C)
      angle_calculated = (angle_calculated_left + angle_calculated_right) / 2
      if (angle_calculated < 55):
        cv2.putText(images[i], "Too deep hip movement, actual angle: " ,(0,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
        cv2.putText(images[i], str(angle_calculated) ,(500,150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
        f.angle_draw(data["RIGHT_HIP"][i,x], data["RIGHT_KNEE"][i,x], data["RIGHT_HIP"][i,y], data["RIGHT_KNEE"][i,y], data["LEFT_HIP"][i,x], data["LEFT_KNEE"][i,x], data["LEFT_HIP"][i,y], data["LEFT_KNEE"][i,y], angle_calculated_right, angle_calculated_left, images[i], width, height)
        """
        dist_ankley_kneey_r = abs(data["RIGHT_HIP"][i,y] - data["RIGHT_HIP"][i,y] )
        dist_anklex_kneex_r = abs(data["RIGHT_HIP"][i,x] - data["RIGHT_KNEE"][i,x])
        dist_ankley_kneey_l = abs(data["LEFT_HIP"][i,y] - data["LEFT_KNEE"][i,y])
        dist_anklex_kneex_l = abs(data["LEFT_HIP"][i,x] - data["LEFT_KNEE"][i,x])
        start_angle_r = math.atan(dist_ankley_kneey_r / dist_anklex_kneex_r)
        start_angle_r = math.degrees(start_angle_r)
        start_angle_l = math.atan(dist_ankley_kneey_l / dist_anklex_kneex_l)
        start_angle_l = math.degrees(start_angle_l)
        if(abs(data["RIGHT_HIP"][i,x] < data["RIGHT_KNEE"][i,x])):
          cv2.ellipse(images[i], (int(data["RIGHT_KNEE"][i,x] * width), int(data["RIGHT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_r + 180 - angle_calculated_right), ((start_angle_r + 180 - angle_calculated_right) + angle_calculated_right), (0, 255, 0), 5)
          cv2.ellipse(images[i], (int(data["LEFT_KNEE"][i,x] * width), int(data["LEFT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_l + 180 - angle_calculated_left), ((start_angle_l + 180 - angle_calculated_left) + angle_calculated_left), (0, 255, 0), 5)
        else:
          cv2.ellipse(images[i], (int(data["RIGHT_KNEE"][i,x] * width), int(data["RIGHT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_r ), ((start_angle_r ) + angle_calculated_right), (0, 255, 0), 5)
          cv2.ellipse(images[i], (int(data["LEFT_KNEE"][i,x] * width), int(data["LEFT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_l ), ((start_angle_l ) + angle_calculated_left), (0, 255, 0), 5)
        """
        deep_hip.append(i)

  #Wrong knee movement
  knee_wrong = []
  for i in range(len(data[keys][:])):
    if(distance_shoulder_heel[i] > 0):
      A = np.array([data["LEFT_FOOT_INDEX"][i,x], data["LEFT_FOOT_INDEX"][i,y]])
      C = np.array([data["LEFT_KNEE"][i,x], data["LEFT_KNEE"][i,y]])
      B = np.array([data["LEFT_ANKLE"][i,x], data["LEFT_ANKLE"][i,y]])
      angle_calculated_left = f.angle_estimation(A, B, C)
      
      A = np.array([data["RIGHT_FOOT_INDEX"][i,x], data["RIGHT_FOOT_INDEX"][i,y]])
      C = np.array([data["RIGHT_KNEE"][i,x], data["RIGHT_KNEE"][i,y]])
      B = np.array([data["RIGHT_ANKLE"][i,x], data["RIGHT_ANKLE"][i,y]])
      angle_calculated_right = f.angle_estimation(A, B, C)
      angle_calculated = (angle_calculated_left + angle_calculated_right) / 2
      if (angle_calculated < 40):
        cv2.putText(images[i], "Too much knee movement, actual angle: " , (0,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
        cv2.putText(images[i], str(angle_calculated), (500,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100,255,100), 2)
        dist_ankley_kneey_r = abs(data["RIGHT_ANKLE"][i,y] - data["RIGHT_FOOT_INDEX"][i,y])
        dist_anklex_kneex_r = abs(data["RIGHT_ANKLE"][i,x] - data["RIGHT_FOOT_INDEX"][i,x])
        dist_ankley_kneey_l = abs(data["LEFT_ANKLE"][i,y] - data["LEFT_FOOT_INDEX"][i,y])
        dist_anklex_kneex_l = abs(data["LEFT_ANKLE"][i,x] - data["LEFT_FOOT_INDEX"][i,x])
        start_angle_r = math.atan(dist_ankley_kneey_r / dist_anklex_kneex_r)
        start_angle_r = math.degrees(start_angle_r)
        start_angle_l = math.atan(dist_ankley_kneey_l / dist_anklex_kneex_l)
        start_angle_l = math.degrees(start_angle_l)
        if(data["RIGHT_FOOT_INDEX"][i,x] > data["RIGHT_ANKLE"][i,x]):
          cv2.ellipse(images[i], (int(data["RIGHT_ANKLE"][i,x] * width), int(data["RIGHT_ANKLE"][i,y] * height)), (25, 25), 0, (start_angle_r - angle_calculated_right), ((start_angle_r - angle_calculated_right) + angle_calculated_right), (0, 255, 0), 3)
          cv2.ellipse(images[i], (int(data["LEFT_ANKLE"][i,x] * width), int(data["LEFT_ANKLE"][i,y] * height)), (25, 25), 0, (start_angle_l - angle_calculated_left), ((start_angle_l - angle_calculated_left) + angle_calculated_left), (0, 255, 0), 3)
        else:
          cv2.ellipse(images[i], (int(data["RIGHT_ANKLE"][i,x] * width), int(data["RIGHT_ANKLE"][i,y] * height)), (25, 25), 0, (start_angle_r + 180 - angle_calculated_right + 20), ((start_angle_r + 180 - angle_calculated_right + 20) + angle_calculated_right), (0, 255, 0), 3)
          cv2.ellipse(images[i], (int(data["LEFT_ANKLE"][i,x] * width), int(data["LEFT_ANKLE"][i,y] * height)), (25, 25), 0, (start_angle_l + 180 - angle_calculated_left + 20), ((start_angle_l + 180 - angle_calculated_left + 20) + angle_calculated_left), (0, 255, 0), 3)
        knee_wrong.append(i)

  #Print in terminal movement evaluation
  knee_errors_percent = 100 - (len(knee_wrong) / (len(data[keys][:]))) *100
  hip_errors_percent = 100 - (len(deep_hip) / (len(data[keys][:]))) * 100
  print("Se tuvo una presición de la rodilla en un %.2f " % (knee_errors_percent))
  print("Se tuvo una presición de la cadera en un %.2f " % (hip_errors_percent))

  file_name = "squat_anotada.mp4"
  f.video_exportation(images,file_name)

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
        input("Error detected")
      if(not (p in knee_wrong or p in deep_hip) and step_brake == 1):
        step_brake = 0
      p += 1
      Video_label.after(20, show)
  show()
"""
def squat_process(path):
  import Functions as f
  import numpy as np
  import math
  import cv2

  x = 0
  y = 0
  
  #Estimation of the pose using mediapipe
  images,width,height,data,keys = f.mediapipe_estimation(path)
  
  print(data.keys)

  #Detection of start pose, if 0 = start
  start_pose = []
  for i in range(len(data[keys][:])):
    a = ((data["RIGHT_HEEL"][i,y] - data["RIGHT_SHOULDER"][i,y]) + (data["LEFT_HEEL"][i,y] - data["LEFT_SHOULDER"][i,y]) / 2)
    start_pose.append(a)
  holder = min(start_pose)
  for i in range(len(data[keys][:])):
    if (abs(holder - start_pose[i]) > 0.2):
      start_pose[i] = 0

  #Too deep down hips movement
  deep_hip = []
  for i in range(len(data[keys][:])):
    if(start_pose[i] > 0):
      A = np.array([data["LEFT_HIP"][i,x], data["LEFT_HIP"][i,y]])
      B = np.array([data["LEFT_KNEE"][i,x], data["LEFT_KNEE"][i,y]])
      C = np.array([data["LEFT_ANKLE"][i,x], data["LEFT_ANKLE"][i,y]])
      angle_calculated_left = f.angle_estimation(A, B, C)
      
      A = np.array([data["RIGHT_HIP"][i,x], data["RIGHT_HIP"][i,y]])
      B = np.array([data["RIGHT_KNEE"][i,x], data["RIGHT_KNEE"][i,y]])
      C = np.array([data["RIGHT_ANKLE"][i,x], data["RIGHT_ANKLE"][i,y]])
      angle_calculated_right = f.angle_estimation(A, B, C)

      angle_calculated = (angle_calculated_left + angle_calculated_right) / 2
      if (angle_calculated < 55):
        
        dist_ankley_kneey_r = abs(data["RIGHT_HIP"][i,y] - data["RIGHT_KNEE"][i,y])
        dist_anklex_kneex_r = abs(data["RIGHT_HIP"][i,x] - data["RIGHT_KNEE"][i,x])
        dist_ankley_kneey_l = abs(data["LEFT_HIP"][i,y] - data["LEFT_KNEE"][i,y])
        dist_anklex_kneex_l = abs(data["LEFT_HIP"][i,x] - data["LEFT_KNEE"][i,x])
        start_angle_r = math.atan(dist_ankley_kneey_r / dist_anklex_kneex_r)
        start_angle_r = math.degrees(start_angle_r)
        start_angle_l = math.atan(dist_ankley_kneey_l / dist_anklex_kneex_l)
        start_angle_l = math.degrees(start_angle_l)
        if(abs(data["RIGHT_HIP"][i,x] < data["RIGHT_KNEE"][i,x])):
          cv2.ellipse(images[i], (int(data["RIGHT_KNEE"][i,x] * width), int(data["RIGHT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_r + 180 - angle_calculated_right), ((start_angle_r + 180 - angle_calculated_right) + angle_calculated_right), (0, 255, 0), 5)
          cv2.ellipse(images[i], (int(data["LEFT_KNEE"][i,x] * width), int(data["LEFT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_l + 180 - angle_calculated_left), ((start_angle_l + 180 - angle_calculated_left) + angle_calculated_left), (0, 255, 0), 5)
        else:
          cv2.ellipse(images[i], (int(data["RIGHT_KNEE"][i,x] * width), int(data["RIGHT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_r ), ((start_angle_r ) + angle_calculated_right), (0, 255, 0), 5)
          cv2.ellipse(images[i], (int(data["LEFT_KNEE"][i,x] * width), int(data["LEFT_KNEE"][i,y] * height)), (25, 25), 0, (start_angle_l ), ((start_angle_l ) + angle_calculated_left), (0, 255, 0), 5)
        deep_hip.append(i)


  file_name = "Squat_procesado.mp4"
  f.video_exportation(images,file_name)
  """