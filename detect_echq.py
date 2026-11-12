import cv2
import numpy as np


def detect_and_rectify_chessboard(frame):
    # Convertir l'image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Appliquer un filtre de flou gaussien pour réduire le bruit
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Détecter les contours avec Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Dilater les contours pour combler les lacunes
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Trouver les contours extérieurs
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Trouver le plus grand contour (l'échiquier)
    chessboard_contour = max(contours, key=cv2.contourArea)

    # Approximer le contour pour obtenir un polygone
    epsilon = 0.02 * cv2.arcLength(chessboard_contour, True)
    approx = cv2.approxPolyDP(chessboard_contour, epsilon, True)

    # Vérifier si le contour a 4 sommets (forme quadrilatérale)
    if len(approx) != 4:
        print(f"Le nombre de sommets détectés est {len(approx)}, au lieu de 4")
        # Si ce n'est pas un quadrilatère, on tente une approche alternative
        rect = cv2.minAreaRect(chessboard_contour)
        approx = cv2.boxPoints(rect).astype(np.int32)

    # Réordonner les points dans l'ordre [haut-gauche, haut-droit, bas-droit, bas-gauche]
    approx = reorder_points(approx)

    # Définir les dimensions de l'échiquier rectifié
    width = height = 800  # Taille de l'échiquier rectifié

    # Définir les points de destination
    dst_points = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)

    # Calculer la matrice de transformation perspective
    matrix = cv2.getPerspectiveTransform(approx.astype(np.float32), dst_points)

    # Appliquer la transformation perspective
    warped = cv2.warpPerspective(frame, matrix, (width, height))

    return warped


def reorder_points(points):
  
    # Convertir en format plus manipulable
    points = points.reshape(4, 2)

    # Calculer la somme des coordonnées x et y pour chaque point
    s = points.sum(axis=1)

    # Le point en haut à gauche aura la somme minimale
    # Le point en bas à droite aura la somme maximale
    tl = points[np.argmin(s)]
    br = points[np.argmax(s)]

    # Calculer la différence des coordonnées
    diff = np.diff(points, axis=1)

    # Le point en haut à droite aura la différence minimale
    # Le point en bas à gauche aura la différence maximale
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]

    # Retourner les points dans l'ordre
    return np.array([tl, tr, br, bl], dtype=np.float32)


def detect_chess_grid(warped):

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    # Appliquer un traitement adaptatif pour améliorer le contraste
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Détecter les lignes
    lines = cv2.HoughLinesP(
        cv2.bitwise_not(adaptive),
        1,
        np.pi / 180,
        threshold=100,
        minLineLength=100,
        maxLineGap=10
    )

    # Dessiner les lignes détectées
    result = warped.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return result


def process_video(video_path):
    """
    Traite la vidéo pour détecter et rectifier l'échiquier

    Args:
        video_path: Chemin vers la vidéo
    """
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo est ouverte
    if not cap.isOpened():
        print(f"Impossible d'ouvrir la vidéo {video_path}")
        return

    # Lire la première image
    ret, frame = cap.read()
    if not ret:
        print("Erreur lors de la lecture de la vidéo")
        return

    # Détecter et rectifier l'échiquier
    warped = detect_and_rectify_chessboard(frame)
    output_filename = "echiquier.png"
    cv2.imwrite(output_filename, warped)
    print(f"Image enregistrée sous : {output_filename}")



    # Détecter la grille de l'échiquier
    grid_detected = detect_chess_grid(warped)

    # Afficher les résultats
    cv2.imshow("Image originale", frame)
    cv2.imshow("Échiquier rectifié", warped)
    cv2.imshow("Grille détectée", grid_detected)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    video_path = r"echiquier_vide_1.mp4"
    process_video(video_path)