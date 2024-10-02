import cv2
import numpy as np
from scipy.signal import butter
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time
from tkinter.filedialog import askopenfilename
import os
from os.path import isfile, join
import sys

def ComputeLaplacianPyramid(frame, max_levels):
    G = frame.copy()
    gpA = [G]
    for i in range(max_levels):
        G = cv2.pyrDown(G)
        gpA.append(G)
    
    lpA = [gpA[-1]]
    for i in range(len(gpA)-1, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.resize(GE, size)
        L = cv2.subtract(gpA[i-1], GE)
        lpA.append(L)
    
    lpA.reverse()
    return lpA

def ComputeRieszPyramid(frame, max_levels):
    laplacian_pyramid = ComputeLaplacianPyramid(frame, max_levels)
    number_of_levels = len(laplacian_pyramid) - 1

    kernel_x = np.array([[0.0, 0.0, 0.0],
                         [0.5, 0.0, -0.5],
                         [0.0, 0.0, 0.0]], dtype=np.float32)

    kernel_y = np.array([[0.0, 0.5, 0.0],
                         [0.0, 0.0, 0.0],
                         [0.0, -0.5, 0.0]], dtype=np.float32)

    riesz_x = []
    riesz_y = []

    for k in range(number_of_levels):
        rx = cv2.filter2D(laplacian_pyramid[k], -1, kernel_x, borderType=cv2.BORDER_REFLECT)
        ry = cv2.filter2D(laplacian_pyramid[k], -1, kernel_y, borderType=cv2.BORDER_REFLECT)
        riesz_x.append(rx)
        riesz_y.append(ry)

    return laplacian_pyramid, riesz_x, riesz_y

def ComputePhaseDifferenceAndAmplitude(current_real, current_x, current_y, previous_real, previous_x, previous_y):
    q_conj_prod_real = current_real * previous_real + current_x * previous_x + current_y * previous_y
    q_conj_prod_x = -current_real * previous_x + previous_real * current_x
    q_conj_prod_y = -current_real * previous_y + previous_real * current_y

    q_conj_prod_amplitude = np.sqrt(q_conj_prod_real ** 2 + q_conj_prod_x ** 2 + q_conj_prod_y ** 2) + 1e-8

    phase_difference = np.arccos(np.clip(q_conj_prod_real / q_conj_prod_amplitude, -1, 1))

    denom_orientation = np.sqrt(q_conj_prod_x ** 2 + q_conj_prod_y ** 2) + 1e-8
    cos_orientation = q_conj_prod_x / denom_orientation
    sin_orientation = q_conj_prod_y / denom_orientation

    phase_difference_cos = phase_difference * cos_orientation
    phase_difference_sin = phase_difference * sin_orientation

    amplitude = np.sqrt(q_conj_prod_amplitude)

    return phase_difference_cos, phase_difference_sin, amplitude

def IIRTemporalFilter(B, A, phase, register0, register1):
    temporally_filtered_phase = B[0] * phase + register0
    register0_new = B[1] * phase + register1 - A[1] * temporally_filtered_phase
    register1_new = B[2] * phase - A[2] * temporally_filtered_phase

    return temporally_filtered_phase, register0_new, register1_new

def AmplitudeWeightedBlur(temporally_filtered_phase, amplitude, blur_kernel):
    numerator = cv2.filter2D(temporally_filtered_phase * amplitude, -1, blur_kernel, borderType=cv2.BORDER_REFLECT)
    denominator = cv2.filter2D(amplitude, -1, blur_kernel, borderType=cv2.BORDER_REFLECT) + 1e-8
    spatially_smooth_temporally_filtered_phase = numerator / denominator

    return spatially_smooth_temporally_filtered_phase

def PhaseShiftCoefficientRealPart(riesz_real, riesz_x, riesz_y, phase_cos, phase_sin):
    phase_magnitude = np.sqrt(phase_cos ** 2 + phase_sin ** 2) + 1e-8
    exp_phase_real = np.cos(phase_magnitude)
    sin_phase_magnitude = np.sin(phase_magnitude)
    exp_phase_x = phase_cos / phase_magnitude * sin_phase_magnitude
    exp_phase_y = phase_sin / phase_magnitude * sin_phase_magnitude

    result = exp_phase_real * riesz_real - exp_phase_x * riesz_x - exp_phase_y * riesz_y

    return result

def CollapseLaplacianPyramid(pyramid):
    current = pyramid[-1]
    for level in reversed(pyramid[:-1]):
        upsampled = cv2.pyrUp(current)
        size = (level.shape[1], level.shape[0])
        upsampled = cv2.resize(upsampled, size)
        current = upsampled + level

    return current

def OnlineRieszVideoMagnification(amplification_factor, low_cutoff, high_cutoff, input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    nyquist_frequency = fps / 2.0
    temporal_filter_order = 1
    low = low_cutoff / nyquist_frequency
    high = high_cutoff / nyquist_frequency
    B, A = butter(temporal_filter_order, [low, high], btype='bandpass')
    B = B.astype(np.float32)
    A = A.astype(np.float32)

    gaussian_kernel_sd = 2
    gaussian_kernel_size = int(gaussian_kernel_sd * 6 + 1)
    gaussian_kernel_1d = cv2.getGaussianKernel(gaussian_kernel_size, gaussian_kernel_sd)
    gaussian_kernel_2d = gaussian_kernel_1d * gaussian_kernel_1d.T

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    ret, frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # Remove grayscale conversion to retain color
    frame = frame.astype(np.float32) / 255.0

    max_levels = 4  # Adjust based on your needs
    previous_laplacian_pyramid, previous_riesz_x, previous_riesz_y = ComputeRieszPyramid(frame, max_levels)

    number_of_levels = len(previous_laplacian_pyramid) - 1

    # Initialize phase and registers with the appropriate shape (including channels)
    phase_cos = [{} for _ in range(number_of_levels)]
    phase_sin = [{} for _ in range(number_of_levels)]
    register0_cos = [{} for _ in range(number_of_levels)]
    register1_cos = [{} for _ in range(number_of_levels)]
    register0_sin = [{} for _ in range(number_of_levels)]
    register1_sin = [{} for _ in range(number_of_levels)]

    for k in range(number_of_levels):
        size = previous_laplacian_pyramid[k].shape
        phase_cos[k] = np.zeros(size, dtype=np.float32)
        phase_sin[k] = np.zeros(size, dtype=np.float32)
        register0_cos[k] = np.zeros(size, dtype=np.float32)
        register1_cos[k] = np.zeros(size, dtype=np.float32)
        register0_sin[k] = np.zeros(size, dtype=np.float32)
        register1_sin[k] = np.zeros(size, dtype=np.float32)

    # Initialize the progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames - 1, desc='Processing frames', unit='frame', ncols=80)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Remove grayscale conversion to retain color
        current_frame = frame.astype(np.float32) / 255.0
        current_laplacian_pyramid, current_riesz_x, current_riesz_y = ComputeRieszPyramid(current_frame, max_levels)

        motion_magnified_laplacian_pyramid = [None] * len(current_laplacian_pyramid)

        # Prepare data for parallel processing
        args_list = []
        for k in range(number_of_levels):
            args_list.append((
                k,
                current_laplacian_pyramid[k],
                current_riesz_x[k],
                current_riesz_y[k],
                previous_laplacian_pyramid[k],
                previous_riesz_x[k],
                previous_riesz_y[k],
                phase_cos[k],
                phase_sin[k],
                register0_cos[k],
                register1_cos[k],
                register0_sin[k],
                register1_sin[k],
                B, A,
                amplification_factor,
                gaussian_kernel_2d
            ))

        # Use ThreadPoolExecutor to parallelize the processing of pyramid levels
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_pyramid_level, args_list))

        # Collect results
        for res in results:
            k = res['level']
            motion_magnified_laplacian_pyramid[k] = res['motion_magnified_coeff']
            # Update phase and registers
            phase_cos[k] = res['phase_cos']
            phase_sin[k] = res['phase_sin']
            register0_cos[k] = res['register0_cos']
            register1_cos[k] = res['register1_cos']
            register0_sin[k] = res['register0_sin']
            register1_sin[k] = res['register1_sin']

        motion_magnified_laplacian_pyramid[number_of_levels] = current_laplacian_pyramid[number_of_levels]
        motion_magnified_frame = CollapseLaplacianPyramid(motion_magnified_laplacian_pyramid)
        motion_magnified_frame = np.clip(motion_magnified_frame, 0, 1)
        motion_magnified_frame_uint8 = (motion_magnified_frame * 255).astype(np.uint8)

        # Write the color frame directly
        out.write(motion_magnified_frame_uint8)

        previous_laplacian_pyramid = current_laplacian_pyramid
        previous_riesz_x = current_riesz_x
        previous_riesz_y = current_riesz_y

        pbar.update(1)  # Update the progress bar

    pbar.close()  # Close the progress bar
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def process_pyramid_level(args):
    (
        k,
        current_laplacian,
        current_riesz_x,
        current_riesz_y,
        previous_laplacian,
        previous_riesz_x,
        previous_riesz_y,
        phase_cos_k,
        phase_sin_k,
        register0_cos_k,
        register1_cos_k,
        register0_sin_k,
        register1_sin_k,
        B, A,
        amplification_factor,
        gaussian_kernel_2d
    ) = args

    phase_difference_cos, phase_difference_sin, amplitude = ComputePhaseDifferenceAndAmplitude(
        current_laplacian,
        current_riesz_x,
        current_riesz_y,
        previous_laplacian,
        previous_riesz_x,
        previous_riesz_y
    )

    phase_cos_k += phase_difference_cos
    phase_sin_k += phase_difference_sin

    phase_filtered_cos, register0_cos_k, register1_cos_k = IIRTemporalFilter(B, A, phase_cos_k, register0_cos_k, register1_cos_k)
    phase_filtered_sin, register0_sin_k, register1_sin_k = IIRTemporalFilter(B, A, phase_sin_k, register0_sin_k, register1_sin_k)

    phase_filtered_cos = AmplitudeWeightedBlur(phase_filtered_cos, amplitude, gaussian_kernel_2d)
    phase_filtered_sin = AmplitudeWeightedBlur(phase_filtered_sin, amplitude, gaussian_kernel_2d)

    phase_magnified_filtered_cos = amplification_factor * phase_filtered_cos
    phase_magnified_filtered_sin = amplification_factor * phase_filtered_sin

    result = PhaseShiftCoefficientRealPart(
        current_laplacian,
        current_riesz_x,
        current_riesz_y,
        phase_magnified_filtered_cos,
        phase_magnified_filtered_sin
    )

    return {
        'level': k,
        'motion_magnified_coeff': result,
        'phase_cos': phase_cos_k,
        'phase_sin': phase_sin_k,
        'register0_cos': register0_cos_k,
        'register1_cos': register1_cos_k,
        'register0_sin': register0_sin_k,
        'register1_sin': register1_sin_k
    }

if __name__ == "__main__":
    amplification_factor = 20  # Adjust as needed
    low_cutoff = 0.4  # In Hz
    high_cutoff = 3.0  # In Hz

    input_video_path = askopenfilename(title="Select the first video file")
    if not input_video_path:
        print("No first video file selected.")
        sys.exit(1)

    _, tail = os.path.split(input_video_path)
    tail = os.path.splitext(tail)[0]
    try:
      fullIndex = tail.index('full')+4
    except:
      fullIndex = 0
    tail = tail[fullIndex:]

    i = 0
    while os.path.exists(f"amp_{tail}_{i}.mp4"):
      i += 1
    output_video_path = f"amp_{tail}_{i}.mp4"
    print(output_video_path)

    start_time = time.time()

    OnlineRieszVideoMagnification(
        amplification_factor,
        low_cutoff,
        high_cutoff,
        input_video_path,
        output_video_path
    )

    end_time = time.time()

    # Calculate the execution time
    execution_time = end_time - start_time

    # Print the result
    print(f"Script executed in: {execution_time:.2f} seconds")
