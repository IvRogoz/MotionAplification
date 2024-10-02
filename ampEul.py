import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import math

def build_laplacian_pyramid(frame, levels):
    pyramid = []
    current_frame = frame.copy()
    for _ in range(levels):
        down = cv2.pyrDown(current_frame)
        up = cv2.pyrUp(down, dstsize=(current_frame.shape[1], current_frame.shape[0]))
        laplacian = current_frame - up
        pyramid.append(laplacian)
        current_frame = down
    pyramid.append(current_frame)
    return pyramid

def reconstruct_from_laplacian_pyramid(pyramid):
    current_frame = pyramid[-1]
    for laplacian in reversed(pyramid[:-1]):
        up = cv2.pyrUp(current_frame, dstsize=(laplacian.shape[1], laplacian.shape[0]))
        current_frame = up + laplacian
    return current_frame

def temporal_bandpass_filter(signal, fps, low, high, order=1):
    nyquist = 0.5 * fps
    low_cut = low / nyquist
    high_cut = high / nyquist
    b, a = butter(order, [low_cut, high_cut], btype='bandpass')
    filtered_signal = filtfilt(b, a, signal, axis=0, padlen=0)
    return filtered_signal

def eulerian_video_magnification(input_video_path, output_video_path, amplification_factor, low_cutoff, high_cutoff, fps=None, levels=4, chrom_attenuation=0.1, chunk_size=60, overlap=30):
    # Read the video properties
    cap = cv2.VideoCapture(input_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) if fps is None else fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps_video, (width, height))

    # Calculate number of chunks
    step = chunk_size - overlap
    num_chunks = max(1, math.ceil((total_frames - overlap) / step))

    # Progress bar for chunks
    for chunk_index in tqdm(range(num_chunks), desc='Processing chunks', unit='chunk', ncols=80):
        start_frame = chunk_index * step
        end_frame = min(start_frame + chunk_size, total_frames)
        actual_chunk_size = end_frame - start_frame

        # Read frames for the current chunk
        cap = cv2.VideoCapture(input_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for _ in range(actual_chunk_size):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        num_frames_in_chunk = len(frames)
        frames = np.array(frames)

        # Build Laplacian pyramids
        pyramids = [build_laplacian_pyramid(frames[i], levels) for i in range(num_frames_in_chunk)]

        # Initialize list to store filtered levels
        filtered_levels = [None] * levels

        # Process each level separately with a progress bar
        for level in tqdm(range(levels), desc=f'Chunk {chunk_index + 1}/{num_chunks} - Processing levels', unit='level', leave=False, ncols=80):
            # Collect the images at this level across all frames
            level_images = [pyramid[level] for pyramid in pyramids]  # list of arrays
            # Stack them into a NumPy array
            level_images = np.stack(level_images, axis=0)  # shape: (num_frames_in_chunk, h, w, c)
            # Reshape for filtering
            num_frames, h, w, c = level_images.shape
            level_data_reshaped = level_images.reshape(num_frames, -1)
            # Apply temporal filtering
            filtered_level_data = temporal_bandpass_filter(level_data_reshaped, fps_video, low_cutoff, high_cutoff)
            # Reshape back to original
            filtered_level_data = filtered_level_data.reshape((num_frames, h, w, c))
            # Amplify
            amplification = amplification_factor
            if level == levels - 1:
                amplification *= chrom_attenuation
            filtered_level_data *= amplification
            filtered_levels[level] = filtered_level_data

        # Reconstruct frames with a progress bar
        output_frames = []
        for i in tqdm(range(num_frames_in_chunk), desc=f'Chunk {chunk_index + 1}/{num_chunks} - Reconstructing frames', unit='frame', leave=False, ncols=80):
            amplified_pyramid = []
            for level in range(levels):
                amplified_laplacian = pyramids[i][level] + filtered_levels[level][i]
                amplified_pyramid.append(amplified_laplacian)
            amplified_pyramid.append(pyramids[i][-1])  # Add the smallest image
            frame = reconstruct_from_laplacian_pyramid(amplified_pyramid)
            frame = np.clip(frame, 0, 1)
            output_frames.append((frame * 255).astype(np.uint8))

        # Write frames to output video, avoiding overlap frames except for the first chunk
        start_write = overlap if chunk_index > 0 else 0
        for i in range(start_write, num_frames_in_chunk):
            out.write(output_frames[i])

    out.release()

if __name__ == '__main__':
    input_video_path = 'input_EDIT.mp4'  # Replace with your input video
    output_video_path = 'output_video.mp4'  # Replace with desired output path
    amplification_factor = 20  # Adjust as needed
    low_cutoff = 0.4  # In Hz
    high_cutoff = 3.0  # In Hz
    fps = None  # Use None to default to video's FPS
    levels = 4  # Number of pyramid levels
    chrom_attenuation = 0.1  # Attenuation for color amplification at the highest level
    chunk_size = 60  # Number of frames per chunk
    overlap = 30  # Number of overlapping frames between chunks

    eulerian_video_magnification(
        input_video_path,
        output_video_path,
        amplification_factor,
        low_cutoff,
        high_cutoff,
        fps,
        levels,
        chrom_attenuation,
        chunk_size,
        overlap
    )
