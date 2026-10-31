
import cv2
import queue
import pyttsx3
import threading
import time
from ultralytics import YOLO


# ==========================================
# 1. SPEECH SYSTEM
# ==========================================

speech_queue = queue.Queue(maxsize=3)


def speech_worker():
    engine = None

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 175)

        while True:
            try:
                message = speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                print("Speaking:", message)
                engine.say(message)
                engine.runAndWait()

            except Exception as error:
                print("Speech error:", error)

            finally:
                speech_queue.task_done()

    except Exception as error:
        print("Speech engine error:", error)

    finally:
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass


def speak(message):
    try:
        speech_queue.put_nowait(message)
        print("Queued:", message)

    except queue.Full:
        # Remove one outdated waiting message.
        try:
            speech_queue.get_nowait()
            speech_queue.task_done()
        except queue.Empty:
            pass

        try:
            speech_queue.put_nowait(message)
            print("Queued latest alert:", message)
        except queue.Full:
            print("Speech queue full; alert skipped.")


threading.Thread(
    target=speech_worker,
    daemon=True
).start()


# ==========================================
# 2. LOAD YOLO MODEL AND CAMERA
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")

camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


# ==========================================
# 3. TRACKING AND ALERT SETTINGS
# ==========================================

CONFIDENCE_THRESHOLD = 0.40

OBSTACLE_REPEAT_SECONDS = 8
NORMAL_REPEAT_SECONDS = 15
REAPPEAR_SECONDS = 3
POSITION_ALERT_COOLDOWN = 2

obstacle_objects = {
    "person",
    "chair",
    "couch",
    "dining table",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "bench",
    "suitcase",
    "potted plant"
}

last_spoken = {}
last_position = {}
last_seen = {}
last_position_alert = {}


# ==========================================
# 4. ESTIMATE IMAGE POSITION
# ==========================================

def get_position(center_x, frame_width):

    ratio = center_x / frame_width

    if ratio < 0.20:
        return "far left"

    elif ratio < 0.40:
        return "slightly left"

    elif ratio < 0.60:
        return "center"

    elif ratio < 0.80:
        return "slightly right"

    else:
        return "far right"


# ==========================================
# 5. APPROXIMATE DISTANCE ESTIMATION
# ==========================================

def estimate_distance(box, frame_width, frame_height):
    """
    Estimate apparent object size in the image.

    This is NOT a real-world distance measurement.
    """

    x1, y1, x2, y2 = box

    box_width = max(0, x2 - x1)
    box_height = max(0, y2 - y1)

    box_area = box_width * box_height
    frame_area = frame_width * frame_height

    if frame_area <= 0:
        return "unknown"

    area_ratio = box_area / frame_area

    if area_ratio >= 0.20:
        return "appears near"

    elif area_ratio >= 0.06:
        return "appears at medium distance"

    else:
        return "appears far away"


# ==========================================
# 6. CREATE VOICE ANNOUNCEMENTS
# ==========================================

def create_message(object_name, position, distance):

    if object_name in obstacle_objects:
        return (
            f"Obstacle alert. {object_name}, "
            f"{distance}, {position}."
        )

    return f"{object_name}, {distance}, {position}."


def should_announce(
    track_key,
    object_name,
    position,
    current_time
):

    previous_seen = last_seen.get(track_key, 0)
    previous_spoken = last_spoken.get(track_key, 0)
    previous_position = last_position.get(track_key)

    is_new = (
        previous_seen == 0
        or current_time - previous_seen > REAPPEAR_SECONDS
    )

    position_changed = (
        previous_position is not None
        and previous_position != position
    )

    if object_name in obstacle_objects:
        repeat_interval = OBSTACLE_REPEAT_SECONDS
    else:
        repeat_interval = NORMAL_REPEAT_SECONDS

    repeat_due = (
        current_time - previous_spoken >= repeat_interval
    )

    position_alert_due = (
        current_time
        - last_position_alert.get(track_key, 0)
        >= POSITION_ALERT_COOLDOWN
    )

    if is_new:
        return True, "new"

    if position_changed and position_alert_due:
        return True, "position"

    if repeat_due:
        return True, "repeat"

    return False, None


# ==========================================
# 7. CAMERA AND OBJECT TRACKING LOOP
# ==========================================

if not camera.isOpened():

    print("ERROR: Could not open the camera.")

else:

    print("AI Visual Companion - Object Tracking")
    print("Tracker: ByteTrack")
    print("Approximate distance estimation: Enabled")
    print("Press Q in the camera window to exit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            # Track objects across consecutive frames.
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            result = results[0]

            annotated_frame = result.plot()

            current_time = time.time()
            frame_height, frame_width = frame.shape[:2]

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes

                if boxes.id is not None:
                    track_ids = (
                        boxes.id.int().cpu().tolist()
                    )
                else:
                    track_ids = [None] * len(boxes)

                class_ids = boxes.cls.int().cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                coordinates = boxes.xyxy.cpu().tolist()

                for track_id, class_id, confidence, box in zip(
                    track_ids,
                    class_ids,
                    confidences,
                    coordinates
                ):

                    object_name = model.names[class_id]

                    x1, y1, x2, y2 = box

                    center_x = (x1 + x2) / 2

                    # Estimate horizontal position.
                    position = get_position(
                        center_x,
                        frame_width
                    )

                    # Estimate apparent distance.
                    distance = estimate_distance(
                        box,
                        frame_width,
                        frame_height
                    )

                    # Identify each tracked object.
                    if track_id is not None:
                        track_key = f"id_{track_id}"
                    else:
                        track_key = f"untracked_{object_name}"

                    # Display tracking information and distance.
                    if track_id is not None:
                        info_text = (
                            f"ID {track_id}: {object_name} "
                            f"| {position} | {distance} "
                            f"| {confidence:.2f}"
                        )
                    else:
                        info_text = (
                            f"{object_name} | {position} "
                            f"| {distance} | {confidence:.2f}"
                        )

                    text_y = max(25, int(y1) - 10)

                    cv2.putText(
                        annotated_frame,
                        info_text,
                        (max(5, int(x1)), text_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2
                    )

                    # Only announce detections with tracking IDs.
                    if track_id is None:
                        continue

                    announce, reason = should_announce(
                        track_key,
                        object_name,
                        position,
                        current_time
                    )

                    if announce:

                        # Pass all THREE required arguments.
                        message = create_message(
                            object_name,
                            position,
                            distance
                        )

                        print(
                            f"Track ID {track_id}: {message} "
                            f"| Confidence: {confidence:.2f}"
                        )

                        speak(message)

                        last_spoken[track_key] = current_time
                        last_position[track_key] = position

                        if reason == "position":
                            last_position_alert[track_key] = (
                                current_time
                            )

                    # Update the last-seen time.
                    last_seen[track_key] = current_time

            # Display the annotated camera frame.
            cv2.imshow(
                "AI Visual Companion - Object Tracking",
                annotated_frame
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Stopping the AI Visual Companion...")

    except Exception as error:
        print("Tracking error:", error)

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("AI Visual Companion stopped.")
