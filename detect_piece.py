import cv2
import numpy as np

# Chessboard setup (90-degree counterclockwise rotation)
board_layout = np.array([
    ['r', 'p', ' ', ' ', ' ', ' ', 'P', 'R'],
    ['n', 'p', ' ', ' ', ' ', ' ', 'P', 'N'],
    ['b', 'p', ' ', ' ', ' ', ' ', 'P', 'B'],
    ['k', 'p', ' ', ' ', ' ', ' ', 'P', 'K'],
    ['q', 'p', ' ', ' ', ' ', ' ', 'P', 'Q'],
    ['b', 'p', ' ', ' ', ' ', ' ', 'P', 'B'],
    ['n', 'p', ' ', ' ', ' ', ' ', 'P', 'N'],
    ['r', 'p', ' ', ' ', ' ', ' ', 'P', 'R']
])

# Piece name mappings
piece_descriptions = {
    'R': 'White Rook', 'N': 'White Knight', 'B': 'White Bishop',
    'Q': 'White Queen', 'K': 'White King', 'P': 'White Pawn',
    'r': 'Black Rook', 'n': 'Black Knight', 'b': 'Black Bishop',
    'q': 'Black Queen', 'k': 'Black King', 'p': 'Black Pawn',
    ' ': 'Empty'
}

# Algebraic notation mappings
piece_symbols = {
    'R': 'R', 'N': 'N', 'B': 'B', 'Q': 'Q', 'K': 'K', 'P': '',
    'r': 'R', 'n': 'N', 'b': 'B', 'q': 'Q', 'k': 'K', 'p': ''
}

def coords_to_notation(row, col):
    return f"{chr(97 + row)}{8 - col}"

def generate_move(piece, start_pos, end_pos, board):
    r_start, c_start = start_pos
    r_end, c_end = end_pos
    symbol = piece_symbols[piece]
    destination = coords_to_notation(r_end, c_end)
    capture = 'x' if board[r_end, c_end] != ' ' else ''
    if piece.upper() == 'P' and capture:
        return f"{chr(97 + r_start)}{capture}{destination}"
    elif piece.upper() == 'P':
        return destination
    return f"{symbol}{capture}{destination}"

print("Initial board layout:")
for row in board_layout:
    print(' '.join(row))
print("=" * 50)

video_feed = cv2.VideoCapture('mat_du_lion.mp4')
reference_feed = cv2.VideoCapture('echiquier_vide_1.mp4')
ret_ref, ref_frame = reference_feed.read()
reference_feed.release()
if not ret_ref:
    raise RuntimeError("Failed to read reference video.")

board_width, board_height = 542 - 241, 327 - 38
cell_width, cell_height = board_width / 8, board_height / 8
source_points = [
    (int(236 + cell_width/2), int(38 + cell_height/2)),
    (int(540 - cell_width/2), int(69 + cell_height/2)),
    (int(187 + cell_width/2), int(327 - cell_height/2)),
    (int(523 - cell_width/2), int(362 - cell_height/2))
]
dest_points = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
transform_matrix = cv2.getPerspectiveTransform(np.float32(source_points), dest_points)
ref_warped = cv2.warpPerspective(ref_frame, transform_matrix, (600, 600))
ref_grayscale = cv2.cvtColor(ref_warped, cv2.COLOR_BGR2GRAY)

GRID_SIZE = 8
BOARD_SIZE = 600
CELL_SIZE = BOARD_SIZE // GRID_SIZE
PADDING = int(CELL_SIZE * 0.15)
MIN_CONTOUR_SIZE = 30
WHITE_PIXEL_RATIO = 0.02
LOWER_HALF_FACTOR = 0.60
MOTION_THRESHOLD = 0.13
CHANGE_THRESHOLD = 0.1

color_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
for i in range(GRID_SIZE):
    for j in range(GRID_SIZE):
        x_start = j * CELL_SIZE + PADDING
        y_start = i * CELL_SIZE + PADDING
        x_end = (j + 1) * CELL_SIZE - PADDING
        y_end = (i + 1) * CELL_SIZE - PADDING
        region = ref_grayscale[y_start:y_end, x_start:x_end]
        if region.size:
            avg_intensity = np.mean(region)
            color_grid[i, j] = 1 if avg_intensity > 150 else 2 if avg_intensity < 100 else 0

ret, current_frame = video_feed.read()
if not ret:
    raise RuntimeError("Failed to read main video.")
current_warped = cv2.warpPerspective(current_frame, transform_matrix, (600, 600))
current_grayscale = cv2.cvtColor(current_warped, cv2.COLOR_BGR2GRAY)
status = 'idle'
is_paused = False
tracked_grids = []
last_avg_grid = None

while True:
    if is_paused:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('p'):
            is_paused = False
            print("Resuming playback")
        elif key == ord('q'):
            break
        continue

    ret, frame = video_feed.read()
    if not ret:
        break
    board_view = cv2.warpPerspective(frame, transform_matrix, (600, 600))
    gray_view = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)

    binary_view = np.zeros_like(board_view)
    cv2.putText(board_view, f"Mode: {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 200, 100) if status == 'idle' else (0, 165, 255), 2)
    if is_paused:
        cv2.putText(board_view, "Halted", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    for i in range(GRID_SIZE + 1):
        cv2.line(board_view, (0, i * CELL_SIZE), (BOARD_SIZE, i * CELL_SIZE), (144, 238, 144), 1)
        cv2.line(board_view, (i * CELL_SIZE, 0), (i * CELL_SIZE, BOARD_SIZE), (144, 238, 144), 1)
        cv2.line(binary_view, (0, i * CELL_SIZE), (BOARD_SIZE, i * CELL_SIZE), (144, 238, 144), 1)
        cv2.line(binary_view, (i * CELL_SIZE, 0), (i * CELL_SIZE, BOARD_SIZE), (144, 238, 144), 1)

    flow = cv2.calcOpticalFlowFarneback(current_grayscale, gray_view, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    midpoint = gray_view.shape[1] // 2
    mag_left, _ = cv2.cartToPolar(flow[:, :midpoint, 0], flow[:, :midpoint, 1])
    mag_right, _ = cv2.cartToPolar(flow[:, midpoint:, 0], flow[:, midpoint:, 1])
    is_moving = max(np.mean(mag_left), np.mean(mag_right)) > MOTION_THRESHOLD and abs(np.mean(mag_left) - np.mean(mag_right)) > 0.023

    detection_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
    if status == 'idle':
        binary = cv2.adaptiveThreshold(gray_view, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x_start = j * CELL_SIZE + PADDING
                y_start = i * CELL_SIZE + PADDING
                x_end = (j + 1) * CELL_SIZE - PADDING
                y_end = (i + 1) * CELL_SIZE - PADDING
                region = cleaned_binary[y_start:y_end, x_start:x_end]
                if region.size:
                    white_ratio = np.count_nonzero(region) / region.size
                    bottom_ratio = np.count_nonzero(region[region.shape[0]//2:]) / region[region.shape[0]//2:].size
                    detected = white_ratio > WHITE_PIXEL_RATIO and (white_ratio >= LOWER_HALF_FACTOR * bottom_ratio or white_ratio >= bottom_ratio)
                    detection_grid[i, j] = 1 if detected else 0
        tracked_grids.append(detection_grid.copy())
        binary_view = cv2.cvtColor(cleaned_binary, cv2.COLOR_GRAY2BGR)

    if status == 'idle' and is_moving:
        status = 'active'
        if tracked_grids:
            avg_grid = np.mean(tracked_grids, axis=0)
            if last_avg_grid is not None:
                delta_grid = avg_grid - last_avg_grid
                min_change, max_change = np.min(delta_grid), np.max(delta_grid)
                if min_change < -CHANGE_THRESHOLD and max_change > CHANGE_THRESHOLD:
                    r_start, c_start = np.unravel_index(np.argmin(delta_grid), delta_grid.shape)
                    r_end, c_end = np.unravel_index(np.argmax(delta_grid), delta_grid.shape)
                    piece = board_layout[r_start, c_start]
                    if piece != ' ':
                        move = generate_move(piece, (r_start, c_start), (r_end, c_end), board_layout)
                        board_layout[r_end, c_end] = piece
                        board_layout[r_start, c_start] = ' '
            last_avg_grid = avg_grid.copy()
        tracked_grids.clear()
    elif status == 'active' and not is_moving:
        status = 'idle'

    if status == 'idle':
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x_start = j * CELL_SIZE + PADDING
                y_start = i * CELL_SIZE + PADDING
                x_end = (j + 1) * CELL_SIZE - PADDING
                y_end = (i + 1) * CELL_SIZE - PADDING
                cx = j * CELL_SIZE + CELL_SIZE // 2
                cy = i * CELL_SIZE + CELL_SIZE // 2
                roi_binary = cleaned_binary[y_start:y_end, x_start:x_end]
                wr = np.count_nonzero(roi_binary) / roi_binary.size if roi_binary.size else 0
                br = np.count_nonzero(roi_binary[roi_binary.shape[0]//2:]) / roi_binary[roi_binary.shape[0]//2:].size if roi_binary[roi_binary.shape[0]//2:].size else 0
                label = f"W:{wr:.2f} B:{br:.2f}"
                color = (0, 255, 255) if detection_grid[i, j] else (255, 0, 255)
                if color_grid[i, j] == 1:
                    color = (0, 0, 0) if detection_grid[i, j] else (100, 100, 100)
                elif color_grid[i, j] == 2:
                    color = (0, 255, 255) if detection_grid[i, j] else (255, 0, 255)

                cv2.putText(board_view, label, (cx - 30, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                if detection_grid[i, j]:
                    cv2.rectangle(board_view, (x_start, y_start), (x_end, y_end), (0, 100, 255), 1)
                    cv2.circle(board_view, (cx, cy), 5, (0, 100, 0), -1)

    combined_view = np.hstack((board_view, binary_view))
    cv2.imshow('Combined Display', combined_view)

    key = cv2.waitKey(100) & 0xFF
    if key == ord('p'):
        is_paused = True
    elif key == ord('q'):
        break

    current_grayscale = gray_view.copy()

video_feed.release()
cv2.destroyAllWindows()
