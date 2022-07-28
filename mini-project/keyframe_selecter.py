import cv2
import os
import mediapipe as mp
import numpy as np
from tqdm import tqdm

def get_centroid_details(file_path):
    """This function gets pose details from mediapipe

    Args:
        file_path (string): This is the path of the video file
        visualize (bool, optional): Provides a visualization for the pose tracking. Defaults to False.

    Returns:
        dict: This dictionary contains a list of upper body pose estimates
    """
    cap = cv2.VideoCapture(file_path)
    mp_pose = mp.solutions.pose

    rh_centroid_x = ()
    rh_centroid_y = ()

    lh_centroid_x = ()
    lh_centroid_y = ()

    with mp_pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            centroid_results = pose.process(image)

            rh_centroid_x_frame = ()
            rh_centroid_y_frame = ()

            lh_centroid_x_frame = ()
            lh_centroid_y_frame = ()

            right_hand_coords = [16, 18, 20, 22]
            left_hand_coords = [15, 17, 21, 19]

            for coord in right_hand_coords:
                rh_centroid_x_frame = rh_centroid_x_frame + (
                    round(
                        centroid_results.pose_landmarks.landmark[coord].x * image_width,
                        2,
                    ),
                )
                rh_centroid_y_frame = rh_centroid_y_frame + (
                    round(
                        centroid_results.pose_landmarks.landmark[coord].y
                        * image_height,
                        2,
                    ),
                )

            for i in left_hand_coords:
                lh_centroid_x_frame = lh_centroid_x_frame + (
                    round(
                        centroid_results.pose_landmarks.landmark[coord].x * image_width,
                        2,
                    ),
                )
                lh_centroid_y_frame = lh_centroid_y_frame + (
                    round(
                        centroid_results.pose_landmarks.landmark[coord].y
                        * image_height,
                        2,
                    ),
                )

            # calculate centroid by taking the arithmetic mean of coordinates
            rh_centroid_x = rh_centroid_x + (
                round(sum(rh_centroid_x_frame) / len(rh_centroid_x_frame), 2),
            )
            rh_centroid_y = rh_centroid_y + (
                round(sum(rh_centroid_y_frame) / len(rh_centroid_y_frame), 2),
            )

            lh_centroid_x = lh_centroid_x + (
                round(sum(lh_centroid_x_frame) / len(lh_centroid_x_frame), 2),
            )
            lh_centroid_y = lh_centroid_y + (
                round(sum(lh_centroid_y_frame) / len(lh_centroid_y_frame), 2),
            )

    cap.release()

    centroid = {}
    centroid["right_hand_x"] = rh_centroid_x
    centroid["right_hand_y"] = rh_centroid_y
    centroid["left_hand_x"] = lh_centroid_x
    centroid["left_hand_y"] = lh_centroid_y

    return centroid

def get_thresholds(centroid):
    right_x = centroid["right_hand_x"]
    right_y = centroid["right_hand_y"]
    left_x = centroid["left_hand_x"]
    left_y = centroid["left_hand_y"]

    initial_no_frames = len(centroid["right_hand_x"])
    hand_distance = []

    def euc_dist(x1, x2, y1, y2):
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    for i in range(initial_no_frames - 1):
        right_distance = euc_dist(right_x[i], right_x[i + 1], right_y[i], right_y[i + 1])
        left_distance = euc_dist(left_x[i], left_x[i + 1], left_y[i], left_y[i + 1])
        hand_distance.append(round(max(right_distance, left_distance), 2))

    alpha = round(0.1 * max(hand_distance), 2)

    hand_height = []
    for i in range(initial_no_frames):
        hand_height.append(round(min(right_y[i], left_y[i]), 2))

    beta=min(sum(hand_height[:10])/len(hand_height[:10]), sum(hand_height[-10:])/len(hand_height[-10:]))

    return alpha, beta, hand_distance, hand_height

    
def write_images(filepath_list):
    for filename in tqdm(filepath_list):
        video_filename=os.path.join("./videos_phoenix/videos", filename + ".mp4")
        image_folder=video_filename.replace("/videos/", "/images/").replace(".mp4", "")
        _, vid_file_name = os.path.split(video_filename)
        video = cv2.VideoCapture(video_filename)
        centroid = get_centroid_details(video_filename)
        alpha, beta, hand_distance, hand_height = get_thresholds(centroid)
        frame_num = 0
        if not os.path.isdir(image_folder):
            os.mkdir(image_folder)
        while True:
            ret, frame = video.read()
            if not ret:
                break
            if(hand_distance[frame_num - 1] > alpha and frame_num != 0 and hand_height[frame_num - 1] < beta):
                outputImageName = 'img_'+str(frame_num) + '.png'
                outputImagePath = os.path.join(image_folder, outputImageName)
                cv2.imwrite(outputImagePath, frame)
            frame_num += 1


# def write_images(filepath_csv):
    
#     # image_folder=filepath.replace("/videos/", "/images/")
#     # for filename in tqdm(os.listdir(filepath)[:int((np.floor(len(filepath)/2)))]):
#     # for filename in tqdm(os.listdir(filepath))
#     for filename in filepath_csv:
#         file=os.path.join("./videos_phoenix", filename)
#         image_folder=file.replace("/videos/", "/images/")

#         _, vid_file_name = os.path.split(file)
#         video = cv2.VideoCapture(file)
#         centroid = get_centroid_details(file)
#         alpha, beta, hand_distance, hand_height = get_thresholds(centroid)
#         frame_num = 0
#         image_folder1=os.path.join(image_folder, vid_file_name.replace(".mp4", ""))
#         if not os.path.isdir(image_folder1):
#             os.mkdir(image_folder1)
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#             if(hand_distance[frame_num - 1] > alpha and frame_num != 0 and hand_height[frame_num - 1] > beta):
#                 outputImageName = 'img_'+str(frame_num) + '.png'
#                 outputImagePath = os.path.join(image_folder1, outputImageName)
#                 # print(outputImagePath)
#                 cv2.imwrite(outputImagePath, frame)
#             frame_num += 1