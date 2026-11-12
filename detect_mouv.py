import cv2
import numpy as np

def detect_skin(frame):
    """
    Détection améliorée de la peau en utilisant YCrCb (plus robuste à l'éclairage).
    """
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    lower_skin = np.array([0, 135, 85], dtype=np.uint8)
    upper_skin = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

    # Morphologie pour améliorer le masque
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    return skin_mask

def main():
    video_path = 'mat_du_lion.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Erreur: Impossible d'ouvrir la vidéo '{video_path}'")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 30
    hand_area_threshold = 1500  # Augmenté pour éviter le bruit

    hand_detected_left = False
    hand_detected_right = False
    previous_hand_detected_left = False
    previous_hand_detected_right = False
    consecutive_frames_with_hand = 0
    required_frames_for_movement = 3
    movement_detected = False
    frame_count = 0

    print("Démarrage de l'analyse...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        display_frame = frame.copy()
        skin_mask = detect_skin(frame)

        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hand_contours = sorted(
            [cnt for cnt in contours if cv2.contourArea(cnt) > hand_area_threshold],
            key=cv2.contourArea,
            reverse=True
        )[:2]

        height, width = frame.shape[:2]
        center_x = 380
        margin = 30  # Zone tampon autour du centre
        cv2.line(display_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

        hand_detected_left = False
        hand_detected_right = False

        for contour in hand_contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                if cx < center_x - margin:
                    hand_detected_left = True
                elif cx > center_x + margin:
                    hand_detected_right = True

                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(display_frame, (cx, cy), 4, (255, 0, 255), -1)

        if hand_detected_left or hand_detected_right:
            consecutive_frames_with_hand += 1
            if consecutive_frames_with_hand >= required_frames_for_movement and not movement_detected:
                movement_detected = True
                print(f"Frame {frame_count}: MOUVEMENT DÉTECTÉ!")
        else:
            consecutive_frames_with_hand = 0
            movement_detected = False

        cv2.putText(display_frame, f"Frame: {frame_count}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if hand_detected_left:
            cv2.putText(display_frame, "JOUEUR GAUCHE", (10, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not previous_hand_detected_left:
                print(f"Frame {frame_count}: Main détectée à GAUCHE")
        
        if hand_detected_right:
            cv2.putText(display_frame, "JOUEUR DROIT", (width - 200, 55), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not previous_hand_detected_right:
                print(f"Frame {frame_count}: Main détectée à DROITE")

        if movement_detected:
            cv2.putText(display_frame, "MOUVEMENT DETECTE", (width//2 - 150, height - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Détection des mains", display_frame)
        cv2.imshow("Masque de peau", skin_mask)

        previous_hand_detected_left = hand_detected_left
        previous_hand_detected_right = hand_detected_right

        key = cv2.waitKey(frame_delay) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            print("Pause - Appuyez sur une touche pour continuer")
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Fin de l'analyse - {frame_count} frames analysées")

if __name__ == "__main__":
    main()
