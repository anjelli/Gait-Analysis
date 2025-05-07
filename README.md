# University-of-Miami---Gait-Analysis

#Project Overview
This project develops a gait analysis pipeline using MediaPipe Pose. It processes video to extract gait parameters and visualize pose. The pipeline detects landmarks, identifies gait events (heel strike, toe off), calculates metrics, and overlays a stick figure on the video.

#Work Accomplished
Key pipeline components are implemented:
  
  -MediaPipe Pose Estimation: Uses MediaPipe Pose (model_complexity=2) for accurate 33-landmark detection per frame.
  -Landmark Data Handling: Processes and stores landmark coordinates/visibility, maintaining recent frame history.
  -Gait Event Detection: Implements a heuristic method for heel strike and toe off based on vertical landmark movement.
  -#Gait Metric Calculation: Calculates preliminary metrics: 
    ![{9F5E019F-7CA0-4E99-BC71-725816BCA130}](https://github.com/user-attachments/assets/63afd08e-cb67-4394-9835-b6129283bcc4)
  -#Stick Figure Visualization: Draws a stick figure on video frames based on detected landmarks(needs some calibration)
    ![{243A9BF2-9F3E-4A05-A435-DD73D41D1C9C}](https://github.com/user-attachments/assets/fab0640a-d6b9-422b-9519-ddb9692e2aed)
    
    Google Drive link of the current work: https://drive.google.com/file/d/1WYXPqofN0uQz3OQkdUiWqmcUVLKCeK0c/view?usp=sharing
  -Output Generation: Prints metrics to console, saves stick figure video overlay.

#Challenges addressed:

  -Output Video Download: Fixed download issues ("Failed to Fetch") by adding cv2.VideoWriter error handling and using google.colab.files.download().
  -Initial Pose Estimation Accuracy: Improved early frame tracking by increasing model_complexity to 2.

#Current Issues and Future Work:

  -Spatial Metric Calibration: Requires accurate real-world measurement for step/stride length. Current code uses a placeholder; robust calibration is needed.
  -Robust Gait Event Detection: Simple thresholds may lack robustness. Advanced techniques (velocity, angles, ML) could improve accuracy.
  -Handling Occlusions and Multiple People: Pipeline assumes single, visible person; does not handle occlusions or multiple subjects.
  
