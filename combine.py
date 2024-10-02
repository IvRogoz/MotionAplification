import sys
from moviepy.editor import VideoFileClip, clips_array
from tkinter import Tk
from tkinter.filedialog import askopenfilename

def resize_and_pad(clip, target_width, target_height):
    # Compute the scaling factor to fit within target dimensions
    scaling_factor = min(target_width / clip.w, target_height / clip.h)
    # Resize the clip
    clip_resized = clip.resize(scaling_factor)
    # Create a new clip with the target dimensions and place the resized clip in the center
    clip_padded = clip_resized.on_color(size=(target_width, target_height), color=(0, 0, 0), pos=('center', 'center'))
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

    # Load the video files
    video1 = VideoFileClip(video1_path)
    video2 = VideoFileClip(video2_path)

    # Define target dimensions (half of Full HD width)
    target_width = 960
    target_height = 1080

    # Process the videos
    video1_processed = resize_and_pad(video1, target_width, target_height)
    video2_processed = resize_and_pad(video2, target_width, target_height)

    # Combine the videos side by side
    final_clip = clips_array([[video1_processed, video2_processed]])

    # Output the final video
    final_clip.write_videofile("output1.mp4")

if __name__ == "__main__":
    main()
