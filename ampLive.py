import cv2
import numpy as np
from scipy.signal import butter, filtfilt


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
    b, a = butter(order, [low_cut, high_cut], btype="bandpass")
    return filtfilt(b, a, signal, axis=0, padlen=0)


def amplify_color(frame_bgr, color_amplification):
    if color_amplification == 1.0:
        return frame_bgr
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    ycrcb = ycrcb.astype(np.float32)
    ycrcb[..., 1] = np.clip((ycrcb[..., 1] - 0.5) * color_amplification + 0.5, 0, 1)
    ycrcb[..., 2] = np.clip((ycrcb[..., 2] - 0.5) * color_amplification + 0.5, 0, 1)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def process_chunk(
    frames,
    fps,
    amplification_factor,
    low_cutoff,
    high_cutoff,
    levels,
    chrom_attenuation,
    color_amplification,
    filter_order=1,
):
    pyramids = [build_laplacian_pyramid(frame, levels) for frame in frames]

    filtered_levels = [None] * levels
    for level in range(levels):
        level_images = np.stack([pyramid[level] for pyramid in pyramids], axis=0)
        num_frames, height, width, channels = level_images.shape
        reshaped = level_images.reshape(num_frames, -1)
        filtered = temporal_bandpass_filter(
            reshaped,
            fps,
            low_cutoff,
            high_cutoff,
            order=filter_order,
        )
        filtered = filtered.reshape((num_frames, height, width, channels))
        amplification = amplification_factor
        if level == levels - 1:
            amplification *= chrom_attenuation
        filtered_levels[level] = filtered * amplification

    output_frames = []
    for index in range(len(frames)):
        amplified_pyramid = [
            pyramids[index][level] + filtered_levels[level][index]
            for level in range(levels)
        ]
        amplified_pyramid.append(pyramids[index][-1])
        frame = reconstruct_from_laplacian_pyramid(amplified_pyramid)
        frame = np.clip(frame, 0, 1)
        frame = amplify_color(frame, color_amplification)
        output_frames.append((frame * 255).astype(np.uint8))
    return output_frames


def live_eulerian_magnification(
    camera_index=0,
    amplification_factor=20,
    low_cutoff=0.4,
    high_cutoff=3.0,
    levels=4,
    chrom_attenuation=0.1,
    color_amplification=1.5,
    chunk_size=60,
    overlap=30,
    default_fps=30.0,
    window_name="Live Motion + Color Amplification",
):
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera index {camera_index}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = default_fps

    buffer = []
    output_started = False
    stop_requested = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_float = frame.astype(np.float32) / 255.0
        buffer.append(frame_float)

        if len(buffer) < chunk_size:
            warmup_frame = frame.copy()
            cv2.putText(
                warmup_frame,
                "Buffering...",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow(window_name, warmup_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        processed_frames = process_chunk(
            buffer,
            fps,
            amplification_factor,
            low_cutoff,
            high_cutoff,
            levels,
            chrom_attenuation,
            color_amplification,
        )

        start_index = 0 if not output_started else overlap
        for processed in processed_frames[start_index:]:
            cv2.imshow(window_name, processed)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                stop_requested = True
                break

        output_started = True
        buffer = buffer[-overlap:]

        if stop_requested:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_index = 0
    amplification_factor = 20
    low_cutoff = 0.4
    high_cutoff = 3.0
    levels = 4
    chrom_attenuation = 0.2
    color_amplification = 1.5
    chunk_size = 60
    overlap = 30
    default_fps = 30.0

    live_eulerian_magnification(
        camera_index=camera_index,
        amplification_factor=amplification_factor,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        levels=levels,
        chrom_attenuation=chrom_attenuation,
        color_amplification=color_amplification,
        chunk_size=chunk_size,
        overlap=overlap,
        default_fps=default_fps,
    )
