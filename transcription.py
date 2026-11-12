import cv2
import numpy as np


def transcribe_chess_game(video_path, reference_video_path=None):
    """
    Transcrit une partie d'échecs complète à partir d'une vidéo en utilisant
    la notation algébrique.

    Args:
        video_path: Chemin vers la vidéo de la partie d'échecs
        reference_video_path: Chemin vers la vidéo de l'échiquier vide (optionnel)

    Returns:
        Une chaîne de caractères contenant la transcription de la partie en notation algébrique
    """
    # Initialisation de l'échiquier (position standard de départ)
    # Ici, nous utilisons la rotation comme dans le code source
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

    # Dictionnaire pour les symboles des pièces en notation algébrique
    piece_symbols = {
        'R': 'T', 'N': 'C', 'B': 'F', 'Q': 'D', 'K': 'R', 'P': '',  # Pièces blanches (Français)
        'r': 'T', 'n': 'C', 'b': 'F', 'q': 'D', 'k': 'R', 'p': ''  # Pièces noires (Français)
    }

    # Configuration pour la détection de l'échiquier et des mouvements
    # Configuration similaire à celle du code fourni
    if reference_video_path:
        reference_feed = cv2.VideoCapture(reference_video_path)
        ret_ref, ref_frame = reference_feed.read()
        reference_feed.release()
        if not ret_ref:
            raise RuntimeError("Impossible de lire la vidéo de référence.")
    else:
        ref_frame = None

    # Configuration des paramètres de transformation perspective
    # Ces valeurs peuvent être ajustées selon la vidéo
    board_width = 542 - 241
    board_height = 327 - 38
    cell_width = board_width / 8
    cell_height = board_height / 8
    source_points = np.float32([
        (int(236 + cell_width / 2), int(38 + cell_height / 2)),
        (int(540 - cell_width / 2), int(69 + cell_height / 2)),
        (int(187 + cell_width / 2), int(327 - cell_height / 2)),
        (int(523 - cell_width / 2), int(362 - cell_height / 2))
    ])
    dest_points = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    transform_matrix = cv2.getPerspectiveTransform(source_points, dest_points)

    # Paramètres pour la détection des pièces et des mouvements
    GRID_SIZE = 8
    BOARD_SIZE = 600
    CELL_SIZE = BOARD_SIZE // GRID_SIZE
    PADDING = int(CELL_SIZE * 0.15)
    WHITE_PIXEL_RATIO = 0.02
    LOWER_HALF_FACTOR = 0.60
    MOTION_THRESHOLD = 0.13
    CHANGE_THRESHOLD = 0.1

    # Initialisation de la capture vidéo
    video_feed = cv2.VideoCapture(video_path)
    ret, current_frame = video_feed.read()
    if not ret:
        raise RuntimeError("Impossible de lire la vidéo principale.")

    # Transformation perspective de la première image
    current_warped = cv2.warpPerspective(current_frame, transform_matrix, (600, 600))
    current_grayscale = cv2.cvtColor(current_warped, cv2.COLOR_BGR2GRAY)

    # Variables pour le suivi de l'état
    status = 'idle'
    tracked_grids = []
    last_avg_grid = None

    # Liste pour stocker les mouvements détectés
    detected_moves = []

    # Pour déterminer quel joueur est actif
    current_player = "white"  # Les blancs commencent
    move_count = 0  # Compteur pour numéroter les coups

    # Traitement de la vidéo
    while True:
        ret, frame = video_feed.read()
        if not ret:
            break

        # Appliquer la transformation perspective
        board_view = cv2.warpPerspective(frame, transform_matrix, (600, 600))
        gray_view = cv2.cvtColor(board_view, cv2.COLOR_BGR2GRAY)

        # Détecter le mouvement avec le flux optique
        flow = cv2.calcOpticalFlowFarneback(current_grayscale, gray_view, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        midpoint = gray_view.shape[1] // 2
        mag_left, _ = cv2.cartToPolar(flow[:, :midpoint, 0], flow[:, :midpoint, 1])
        mag_right, _ = cv2.cartToPolar(flow[:, midpoint:, 0], flow[:, midpoint:, 1])

        # Déterminer s'il y a un mouvement significatif
        mean_left = np.mean(mag_left)
        mean_right = np.mean(mag_right)
        is_moving = max(mean_left, mean_right) > MOTION_THRESHOLD and abs(mean_left - mean_right) > 0.023

        # Logique pour la détection des coups
        if status == 'idle' and is_moving:
            status = 'active'
            if tracked_grids:
                # Calculer la grille moyenne de détection
                avg_grid = np.mean(tracked_grids, axis=0)

                if last_avg_grid is not None:
                    # Calculer les changements dans la grille
                    delta_grid = avg_grid - last_avg_grid
                    min_change = np.min(delta_grid)
                    max_change = np.max(delta_grid)

                    # Si un changement significatif est détecté
                    if min_change < -CHANGE_THRESHOLD and max_change > CHANGE_THRESHOLD:
                        # Trouver la case de départ et d'arrivée
                        r_start, c_start = np.unravel_index(np.argmin(delta_grid), delta_grid.shape)
                        r_end, c_end = np.unravel_index(np.argmax(delta_grid), delta_grid.shape)

                        # Convertir en notation algébrique
                        start_square = coords_to_notation(r_start, c_start)
                        end_square = coords_to_notation(r_end, c_end)

                        # Identifier la pièce qui a été déplacée
                        piece = board_layout[r_start, c_start]

                        if piece != ' ':
                            # Générer la notation du coup
                            move = generate_move(piece, (r_start, c_start), (r_end, c_end), board_layout)

                            # Déplacer la pièce sur l'échiquier virtuel
                            board_layout[r_end, c_end] = piece
                            board_layout[r_start, c_start] = ' '

                            # Déterminer le joueur qui a effectué le coup
                            player = "white" if piece.isupper() else "black"

                            # Ajouter le coup à la liste des mouvements détectés
                            if player == "white":
                                move_count += 1
                                detected_moves.append((move_count, move, None))
                            else:
                                detected_moves[-1] = (move_count, detected_moves[-1][1], move)

                            # Passer au joueur suivant
                            current_player = "black" if current_player == "white" else "white"

                # Mettre à jour la grille moyenne précédente
                last_avg_grid = avg_grid.copy()

            # Effacer les grilles suivies
            tracked_grids = []

        elif status == 'active' and not is_moving:
            status = 'idle'

        # Analyser l'échiquier en état 'idle'
        if status == 'idle':
            # Utiliser un seuillage adaptatif pour détecter les pièces
            binary = cv2.adaptiveThreshold(gray_view, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 11, 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned_binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

            # Analyser chaque case pour détecter les pièces
            detection_grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)

            for i in range(GRID_SIZE):
                for j in range(GRID_SIZE):
                    x_start = j * CELL_SIZE + PADDING
                    y_start = i * CELL_SIZE + PADDING
                    x_end = (j + 1) * CELL_SIZE - PADDING
                    y_end = (i + 1) * CELL_SIZE - PADDING

                    region = cleaned_binary[y_start:y_end, x_start:x_end]
                    if region.size:
                        white_ratio = np.count_nonzero(region) / region.size

                        # Calculer le ratio de la moitié inférieure (important pour distinguer les pièces)
                        bottom_half = region[region.shape[0] // 2:]
                        bottom_ratio = np.count_nonzero(bottom_half) / bottom_half.size if bottom_half.size else 0

                        # Déterminer si une pièce est présente
                        detected = (white_ratio > WHITE_PIXEL_RATIO) and (
                                white_ratio >= LOWER_HALF_FACTOR * bottom_ratio or white_ratio >= bottom_ratio)

                        detection_grid[i, j] = 1 if detected else 0

            # Ajouter la grille de détection à la liste pour le traitement ultérieur
            tracked_grids.append(detection_grid.copy())

        # Mettre à jour l'image en niveaux de gris courante
        current_grayscale = gray_view.copy()

        # Afficher la vidéo avec informations (optionnel, peut être désactivé pour le traitement)
        cv2.putText(board_view, f"Status: {status}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0) if status == 'idle' else (0, 0, 255), 2)

        cv2.imshow('Chess Game Analysis', board_view)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Libérer les ressources
    video_feed.release()
    cv2.destroyAllWindows()

    # Générer la transcription de la partie
    transcription = format_algebraic_notation(detected_moves)

    return transcription


def coords_to_notation(row, col):
    """
    Convertit les coordonnées de la grille (0-7, 0-7) en notation algébrique (a1-h8).
    """
    column = chr(97 + row)  # a-h
    rank = 8 - col  # 1-8
    return f"{column}{rank}"


def generate_move(piece, start_pos, end_pos, board):
    """
    Génère la notation algébrique pour un mouvement de pièce.
    """
    r_start, c_start = start_pos
    r_end, c_end = end_pos

    # Symboles des pièces en français
    piece_symbols = {
        'R': 'T', 'N': 'C', 'B': 'F', 'Q': 'D', 'K': 'R', 'P': '',  # Pièces blanches
        'r': 't', 'n': 'c', 'b': 'f', 'q': 'd', 'k': 'r', 'p': ''  # Pièces noires
    }

    symbol = piece_symbols[piece]
    destination = coords_to_notation(r_end, c_end)

    # Vérifier s'il s'agit d'une capture
    capture = 'x' if board[r_end, c_end] != ' ' else ''

    # Cas spécial pour les pions qui capturent
    if piece.upper() == 'P' and capture:
        start_col = chr(97 + r_start)  # Colonne de départ (a-h)
        return f"{start_col}{capture}{destination}"
    # Cas normal pour les pions
    elif piece.upper() == 'P':
        return destination
    # Autres pièces
    else:
        return f"{symbol}{capture}{destination}"


def format_algebraic_notation(moves):
    """
    Formate les mouvements détectés en notation algébrique standard.

    Args:
        moves: Liste de tuples (numéro du coup, coup blanc, coup noir)

    Returns:
        Chaîne de caractères formatée avec la notation algébrique
    """
    formatted = ""

    for move_num, white_move, black_move in moves:
        if black_move:
            formatted += f"{move_num}. {white_move} {black_move}\n"
        else:
            formatted += f"{move_num}. {white_move}\n"

    return formatted


# Exemple d'utilisation pour transcrire le mat du berger
def main():
    video_path = "mat_du_lion.mp4"
    reference_path = "echiquier_vide_1.mp4"

    # Transcrire la partie
    transcription = transcribe_chess_game(video_path, reference_path)

    print("\nTranscription de la partie:")
    print(transcription)


if __name__ == "__main__":
    main()
