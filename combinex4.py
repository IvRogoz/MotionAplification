import sys
from moviepy.editor import VideoFileClip, clips_array
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def resize_and_pad(clip, target_width, target_height):
    # Compute the scaling factor to fit within target dimensions
    scaling_factor = min(target_width / clip.w, target_height / clip.h)
    # Resize the clip while maintaining aspect ratio
    clip_resized = clip.resize(scaling_factor)
    # Create a new clip with the target dimensions and place the resized clip in the center
    clip_padded = clip_resized.on_color(
        size=(target_width, target_height),
        color=(0, 0, 0),
        pos=('center', 'center')
    )
    return clip_padded

def main():
    # Initialize Tkinter root
    root = Tk()
    root.withdraw()  # Hide the root window

    # Open file dialog to select the first video file
    video1_path = askopenfilename(title="Select the first video file")
    if not video1_path:
        print("No first video file selected.")
        sys.exit(1)

    # Open file dialog to select the second video file
    video2_path = askopenfilename(title="Select the second video file")
    if not video2_path:
        print("No second video file selected.")
        sys.exit(1)

    # Open file dialog to select the third video file
    video3_path = askopenfilename(title="Select the third video file")
    if not video3_path:
        print("No third video file selected.")
        sys.exit(1)

    # Open file dialog to select the fourth video file
    video4_path = askopenfilename(title="Select the fourth video file")
    if not video4_path:
        print("No fourth video file selected.")
        sys.exit(1)

    # Load the video files
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)
    video3 = VideoFileClip(video3_path)
    video4 = VideoFileClip(video4_path)

    # Define target dimensions for each video (quarter of Full HD)
    target_width = 960   # 1920 / 2
    target_height = 540  # 1080 / 2

    # Process the videos
    video1_processed = resize_and_pad(video1, target_width, target_height)
    video2_processed = resize_and_pad(video2, target_width, target_height)
    video3_processed = resize_and_pad(video3, target_width, target_height)
    video4_processed = resize_and_pad(video4, target_width, target_height)

    # Combine the videos in a 2x2 grid
    final_clip = clips_array([
        [video1_processed, video2_processed],
        [video3_processed, video4_processed]
    ])

    # Output the final video
    final_clip.write_videofile("output.mp4")

if __name__ == "__main__":
    main()
