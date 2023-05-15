def pull_up_ana(path,Video_label,Info_posture,Load_button):
    import cv2
    import mediapipe as mp
    import numpy as np
    import Functions as f
    import math
    from PIL import Image
    from PIL import ImageTk

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

    # Too close hands
    bad_hand_position = []
    for i in range((frames_amount - 1)):
        distance_shoulder = math.sqrt(pow((data["RIGHT_SHOULDER"][i,x] - data["LEFT_SHOULDER"][i,x]),2) + pow((data["RIGHT_SHOULDER"][i,y] - data["LEFT_SHOULDER"][i,y]),2))
        distance_wrist = math.sqrt(pow((data["RIGHT_WRIST"][i,x] - data["LEFT_WRIST"][i,x]),2) + pow((data["RIGHT_WRIST"][i,y] - data["LEFT_WRIST"][i,y]),2))
        if(abs(distance_shoulder - distance_wrist) < 0.01):
            cv2.putText(images[i], "Open more the hands" , (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
            bad_hand_position.append(i)

    #detect start of pull up
    distance_shoulder_wrist = []
    for i in range(frames_amount - 1):
        a = math.sqrt(pow((data["RIGHT_SHOULDER"][i,x] - data["RIGHT_WRIST"][i,x]),2) + pow((data["RIGHT_SHOULDER"][i,y] - data["RIGHT_WRIST"][i,y]),2))
        b = math.sqrt(pow((data["LEFT_SHOULDER"][i,x] - data["LEFT_WRIST"][i,x]),2) + pow((data["LEFT_SHOULDER"][i,y] - data["LEFT_WRIST"][i,y]),2))
        c = (a + b) / 2
        distance_shoulder_wrist.append(c)
    holder = min(distance_shoulder_wrist)
    for i in range(frames_amount - 1):
        if (abs(holder - distance_shoulder_wrist[i]) > 0.01):
            distance_shoulder_wrist[i] = 0

    # Too low lift
    low_lift = []
    for i in range((frames_amount - 1)):
        if(distance_shoulder_wrist[i] > 0):
            lift_estimated = ((data["NOSE"][i,y] - data["RIGHT_WRIST"][i,y]) + (data["NOSE"][i,y] - data["LEFT_WRIST"][i,y]) / 2)
            if(lift_estimated > 0):
                cv2.putText(images[i], "Incomplete lift" , (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,255,100), 2)
                low_lift.append(i)

    #Print in terminal movement evaluation
    hand_error_percent = 100 - (len(bad_hand_position) / frames_amount) * 100
    lift_error_percent = 100 - (len(low_lift) / frames_amount) * 100
    print("Se tuvo una presición de las manos en un %.2f " % (hand_error_percent))
    print("Se tuvo una presición de elevación en un %.2f " % (lift_error_percent))

    #Export video
    file_name = ("Pull_up_anotada.mp4")
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
            if(((p in bad_hand_position) or (p in low_lift)) and step_brake == 0):
                step_brake = 1
                input("Error detected")
            if(not (p in bad_hand_position or p in low_lift) and step_brake == 1):
                step_brake = 0
            p += 1
            Video_label.after(20, show)
    show()