import sys
import os
import argparse
import cv2
import mediapipe as mp
import json
import csv


def dir_path(string):
    """function to check whether the directory path is valid for the save_dir argument

    Args:
        string (string): Directory path

    Raises:
        Exception: Directory not found exception
    """
    if os.path.isdir(string):
        return string
    else:
        raise Exception(f"The given path: {string} is not a directory.")


def vid_path(string):
    """function to check whether the path given is a file for the file_path argument

    Args:
        string (string): File path

    Raises:
        Exception: File not found exception
    """
    if os.path.isfile(string):
        return string
    else:
        raise Exception(f"The given path: {string} is not a file.")


def center_crop(img, dim):
    """Returns center cropped image

    Args:
        img (numpy.ndarray): Image to be cropped
        dim (tuple): Dimensions (width, height) to be cropped

    Returns:
        numpy.ndarray: Cropped image
    """

    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = int(width / 2), int(height / 2)
    cw2, ch2 = int(crop_width / 2), int(crop_height / 2)
    crop_img = img[mid_y - ch2: mid_y + ch2, mid_x - cw2: mid_x + cw2]
    return crop_img


def get_video_details(filename, video):
    """function to get summary details from the filename and opencv

    Args:
        filename (string): Name of the video file
        video (VideoCapture): An opencv videocapture object that contains the video file

    Returns:
        dict: A dictionary that contains video summary details
    """

    # get video properties using the video filename. (use splittext to remove extension)
    vid_name_array = os.path.splitext(filename)[0].split("_")

    env = "home" if vid_name_array[0][1] == "H" else "studio"
    signer_id = vid_name_array[1][1]
    gloss_id = vid_name_array[2]
    position = vid_name_array[3] if env == "home" else "S"

    # get properties of the video using openCV
    num_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))

    # dictionary to store the video summary details
    video_summary = {}

    video_summary["video_path"] = file_path
    video_summary["env"] = env
    video_summary["signer_id"] = signer_id
    video_summary["gloss_id"] = gloss_id
    video_summary["position"] = position
    video_summary["num_of_frames"] = num_of_frames
    video_summary["fps"] = fps
    video_summary["height"] = height
    video_summary["width"] = width
    return video_summary


def crop_video(file_path, filename, save_dir, fps):
    """This function crops the video file into 320x320 and saves it

    Args:
        file_path (string): This is the path of the video file
        filename (string): This is the filename of the video file
        save_dir (string): This is the path to the directory to which the cropped video should be saved
        fps (float): The frames per second of the video

    Returns:
        string: path of the output video file
    """

    video = cv2.VideoCapture(file_path)
    output_path = os.path.join(save_dir, filename.split(".")[0] + ".mp4")
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    output = cv2.VideoWriter(output_path, fourcc, fps, (320, 320))
    while True:
        ret, frame = video.read()
        if not ret:
            break
        crop_img = center_crop(frame, (320, 320))
        output.write(crop_img)
    output.release()

    return output_path


def get_pose_details(file_path):
    """This function gets pose details from mediapipe

    Args:
        file_path (string): This is the path of the video file
        visualize (bool, optional): Provides a visualization for the pose tracking. Defaults to False.

    Returns:
        dict: This dictionary contains a list of upper body pose estimates
    """
    cap = cv2.VideoCapture(file_path)
    mp_pose = mp.solutions.pose

    pose_x = []
    pose_y = []

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

            pose_results = pose.process(image)

            pose_x_frame = []
            pose_y_frame = []

            for i in range(23):
                pose_x_frame.append(
                    round(
                        pose_results.pose_landmarks.landmark[i].x * image_width, 2)
                )
                pose_y_frame.append(
                    round(
                        pose_results.pose_landmarks.landmark[i].y * image_height, 2)
                )
            # print(pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width)

            pose_x.append(pose_x_frame)
            pose_y.append(pose_y_frame)
    cap.release()

    pose_dict = {}
    pose_dict["pose_x"] = pose_x
    pose_dict["pose_y"] = pose_y

    return pose_dict


def get_hand_details(file_path):
    """This function generates the hand tracking with mediapipe functions

    Args:
        file_path (string): This is the path of the video file
        visualize (bool, optional): Provides a visualization for the hand tracking. Defaults to False.

    Returns:
        dict: This dictionary contains 4 keys - hand1_x, hand1_y, hand2_x, hand2_y. The x,y coordinates of each hand
    """
    cap = cv2.VideoCapture(file_path)
    mp_hands = mp.solutions.hands

    hand1_x = []
    hand1_y = []
    hand2_x = []
    hand2_y = []
    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_height, image_width, _ = image.shape

            hand_results = hands.process(image)
            if not hand_results.multi_hand_landmarks:
                continue
            hand1_x_frame = []
            hand1_y_frame = []
            hand2_x_frame = []
            hand2_y_frame = []

            count = 0
            for hand_landmarks in hand_results.multi_hand_landmarks:
                count += 1
                # accept maximum of two hands
                if count > 2:
                    break
                for i in range(21):
                    if count == 1:
                        hand1_x_frame.append(
                            round(
                                hand_landmarks.landmark[i].x * image_width, 2)
                        )
                        hand1_y_frame.append(
                            round(
                                hand_landmarks.landmark[i].y * image_height, 2)
                        )
                    else:
                        hand2_x_frame.append(
                            round(
                                hand_landmarks.landmark[i].x * image_width, 2)
                        )
                        hand2_y_frame.append(
                            round(
                                hand_landmarks.landmark[i].y * image_height, 2)
                        )
                # inserting NaN values for the missing hand coordinates
                if len(hand_results.multi_hand_landmarks) == 1:
                    hand2_x_frame.append(["NaN"] * 21)
                    hand2_y_frame.append(["NaN"] * 21)

            hand1_x.append(hand1_x_frame)
            hand1_y.append(hand1_y_frame)
            hand2_x.append(hand2_x_frame)
            hand2_y.append(hand2_y_frame)
    cap.release()

    hand_dict = {}
    hand_dict["hand1_x"] = hand1_x
    hand_dict["hand1_y"] = hand1_y
    hand_dict["hand2_x"] = hand2_x
    hand_dict["hand2_y"] = hand2_y

    return hand_dict


if __name__ == "__main__":
    # creating a parser object to manage arguments to script and create a cli
    parser = argparse.ArgumentParser(
        description="Script to crop, resize a video and then return the summary, pose estimates, hand keypoints of the video file."
    )

    # required argument for the input video file
    parser.add_argument(
        "--file_path", type=vid_path, help="path of the input video file", required=True
    )

    # argument to take in path of directory to save video and pose files. (defaults to current directory)
    parser.add_argument(
        "--save_dir",
        type=dir_path,
        default="./",
        help="path of directory to save video and pose files. (defaults to current directory)",
    )

    args = parser.parse_args()

    file_path = args.file_path
    save_dir = args.save_dir
    
    # get filename of video
    _, vid_file_name = os.path.split(file_path)
    # get opencv video object
    video = cv2.VideoCapture(file_path)

    # 4 modules for getting metadata, cropping and getting pose, hand details from video
    video_summary = get_video_details(vid_file_name, video)
    rgb_path = crop_video(file_path, vid_file_name,
                          save_dir, video_summary["fps"])
    pose_dict = get_pose_details(file_path)
    hand_dict = get_hand_details(file_path)

    pose_dict.update(hand_dict)

    # Writing to <filename>.json file
    pose_path = os.path.join(
        save_dir, f"{os.path.splitext(vid_file_name)[0]}.json")
    with open(pose_path, "w") as f:
        f.write(json.dumps(pose_dict))

    # adding extra details to the video summary
    video_summary["rgb_path"] = rgb_path
    video_summary["pose_path"] = pose_path

    # printing to stdout
    print("============= VIDEO SUMMARY =============")
    for detail in video_summary:
        print(f'"{detail}": {video_summary[detail]}')
