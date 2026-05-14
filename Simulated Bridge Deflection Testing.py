
"""
Single Marker Deflection Measurement - Y-axis only
Method: Track Y-axis position change (tvec[1])
"""
import cv2
import numpy as np
import sys
import time

# ==============================
# Parameters
# ==============================
ARUCO_ID = 985          # Single marker ID
CELL_SIZE = 1.364       # cm per cell
MARKER_SIDE_CM = CELL_SIZE * 7
MARKER_SIDE_M = MARKER_SIDE_CM / 100
BUFFER_SIZE = 5          # Moving average filter size

VIDEO_URL = "http://192.168.xx.x:xxxx/video"#Your VIDEO_URL
print("=" * 60)
print("Single Marker Deflection Monitor (Y-axis only)")
print("=" * 60)
print(f"Marker ID: {ARUCO_ID}")
print(f"Marker size: {MARKER_SIDE_CM:.3f}cm")
print(f"Tracking: Y-axis (vertical position)")
print(f"Smoothing: {BUFFER_SIZE}-frame moving average")
print()

# ==============================
# OpenCV Compatibility
# ==============================
def get_aruco_dict(dict_id):
    if cv2.__version__[0] >= '4':
        return cv2.aruco.getPredefinedDictionary(dict_id)
    else:
        return cv2.aruco.Dictionary_get(dict_id)

def detect_aruco_markers(gray, aruco_dict, parameters):
    if cv2.__version__[0] >= '4':
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
        return corners, ids, rejected
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters
        )
        return corners, ids, rejected

def estimate_pose(corners, marker_length, camera_matrix, dist_coeffs):
    """Estimate marker pose using solvePnP"""
    obj_pts = np.array([
        [-marker_length/2,  marker_length/2, 0],
        [ marker_length/2,  marker_length/2, 0],
        [ marker_length/2, -marker_length/2, 0],
        [-marker_length/2, -marker_length/2, 0]
    ], dtype=np.float32)

    rvecs, tvecs = [], []
    for corner in corners:
        corner_pts = corner.astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, corner_pts, camera_matrix, dist_coeffs,
            useExtrinsicGuess=False
        )
        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)
    return np.array(rvecs), np.array(tvecs)

# ==============================
# Initialize
# ==============================
print("Connecting to video stream...")
cap = cv2.VideoCapture(VIDEO_URL)
if not cap.isOpened():
    print("ERROR: Cannot connect to video stream!")
    sys.exit(1)

actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"OK: Video stream connected: {actual_width}x{actual_height}")

# Approximate camera matrix
fx, fy = actual_width * 1.2, actual_height * 2.0
cx, cy = actual_width / 2, actual_height / 2
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype=np.float64)
dist_coeffs = np.zeros(5, dtype=np.float64)

# ArUco detector
aruco_dict = get_aruco_dict(cv2.aruco.DICT_ARUCO_ORIGINAL)
if cv2.__version__[0] >= '4':
    aruco_params = cv2.aruco.DetectorParameters()
else:
    aruco_params = cv2.aruco.DetectorParameters_create()

# ==============================
# State Variables
# ==============================
y_ref = None      # Reference Y position (meters)
is_ref_set = False
current_deflection = 0.0
max_deflection = 0.0

# Smoothing filter - Y-axis only
y_buffer = []

# Create window
cv2.namedWindow('Y-Axis Deflection Monitor', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Y-Axis Deflection Monitor', 900, 600)

print("\n" + "=" * 60)
print("Controls:")
print("  [SPACE] - Set reference (marker at rest)")
print("  [r]     - Reset max value")
print("  [s]     - Save screenshot")
print("  [q]     - Quit")
print("=" * 60)
print()

# ==============================
# Main Loop
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Stream interrupted, reconnecting...")
        time.sleep(1)
        continue

    display = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = detect_aruco_markers(gray, aruco_dict, aruco_params)

    if ids is not None:
        detected_ids = ids.flatten()
        if ARUCO_ID in detected_ids:
            idx = np.where(detected_ids == ARUCO_ID)[0][0]
            c = corners[idx]

            cv2.aruco.drawDetectedMarkers(display, [c])

            rvecs, tvecs = estimate_pose(
                [c], MARKER_SIDE_M, camera_matrix, dist_coeffs
            )

            if len(tvecs) > 0:
                rvec = rvecs[0]
                tvec = tvecs[0]

                # Current Y position (meters)
                y_raw = tvec[1, 0]

                # Apply moving average filter
                y_buffer.append(y_raw)
                if len(y_buffer) > BUFFER_SIZE:
                    y_buffer.pop(0)
                y_current = sum(y_buffer) / len(y_buffer)

                # Draw axes
                try:
                    cv2.drawFrameAxes(display, camera_matrix, dist_coeffs,
                                     rvec, tvec, MARKER_SIDE_M * 0.3)
                except:
                    cv2.aruco.drawAxis(display, camera_matrix, dist_coeffs,
                                      rvec, tvec, MARKER_SIDE_M * 0.3)

                # Calculate deflection
                info_lines = []
                info_lines.append(f"Y position: {y_current*100:.2f}cm")
                info_lines.append(f"Raw: {y_raw*100:.2f}cm  (filtered)")

                if is_ref_set:
                    # Y-axis change = deflection
                    current_deflection = (y_current - y_ref) * 100  # to cm

                    info_lines.append(f"Y_ref = {y_ref*100:.2f}cm")
                    info_lines.append(f"Deflection = Y - Y_ref = {current_deflection:.3f}cm")

                    # Update max
                    if abs(current_deflection) > abs(max_deflection):
                        max_deflection = current_deflection
                else:
                    info_lines.append("Press [SPACE] to set reference")

                # Display info on image
                status = "REF SET" if is_ref_set else "Press SPACE for ref"
                color = (0, 255, 0) if is_ref_set else (0, 0, 255)
                cv2.putText(display, status, (20, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                for j, text in enumerate(info_lines):
                    cv2.putText(display, text, (20, 60 + j * 22),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                y_pos = 60 + len(info_lines) * 22 + 15
                if is_ref_set:
                    cv2.putText(display, f"Deflection: {current_deflection:.3f}cm",
                               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_pos += 25

                if max_deflection != 0:
                    cv2.putText(display, f"MAX: {max_deflection:.3f}cm",
                               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(display, f"Marker {ARUCO_ID} not found",
                       (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ids is not None:
                detected = ", ".join(map(str, detected_ids))
                cv2.putText(display, f"Detected: {detected}",
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    else:
        cv2.putText(display, "No markers detected",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Y-Axis Deflection Monitor', display)

    # Key handling
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE: Set reference
        # Clear buffer to get fresh reference
        y_buffer.clear()

        if ids is not None and ARUCO_ID in ids.flatten():
            idx = np.where(ids.flatten() == ARUCO_ID)[0][0]
            c = corners[idx]
            rvecs, tvecs = estimate_pose(
                [c], MARKER_SIDE_M, camera_matrix, dist_coeffs
            )
            if len(tvecs) > 0:
                y_ref = tvecs[0][1, 0]
                is_ref_set = True
                max_deflection = 0.0
                print(f"OK: Reference set. Y_ref={y_ref*100:.2f}cm")
        else:
            print("WARNING: Marker not detected, cannot set reference")

    elif key == ord('r'):  # r: Reset
        max_deflection = 0.0
        y_buffer.clear()
        print("OK: Max value reset")

    elif key == ord('s'):  # s: Screenshot
        filename = f"screenshot_{time.strftime('%H%M%S')}.jpg"
        cv2.imwrite(filename, display)
        print(f"OK: Screenshot saved: {filename}")

    elif key == ord('q'):  # q: Quit
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print(f"Measurement complete! Max deflection: {max_deflection:.3f}cm")
print("=" * 60)
