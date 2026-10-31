
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

# These dictionaries now use tracking IDs as their keys.
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
# 5. CREATE VOICE ANNOUNCEMENTS
# ==========================================

def create_message(object_name, position):

    if object_name in obstacle_objects:
        return (
            f"Obstacle alert. {object_name} "
            f"ahead, {position}."
        )

    return f"{object_name}, {position}."


def should_announce(track_key, object_name,
                    position, current_time):

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
# 6. CAMERA AND OBJECT TRACKING LOOP
# ==========================================

if not camera.isOpened():

    print("ERROR: Could not open the camera.")

else:

    print("AI Visual Companion - Object Tracking")
    print("Tracker: ByteTrack")
    print("Press Q in the camera window to exit.")

    try:

        while True:

            success, frame = camera.read()

            if not success:
                print("Could not read camera frame.")
                break

            # TRACK objects across consecutive frames.
            # persist=True preserves tracker state.
            results = model.track(
                frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=640,
                conf=CONFIDENCE_THRESHOLD,
                verbose=False
            )

            result = results[0]

            # Draw YOLO boxes and labels.
            annotated_frame = result.plot()

            current_time = time.time()
            frame_height, frame_width = frame.shape[:2]

            if result.boxes is not None and len(result.boxes) > 0:

                boxes = result.boxes

                # Read tracking IDs when the tracker supplies them.
                if boxes.id is not None:

                    track_ids = (
                        boxes.id.int().cpu().tolist()
                    )

                else:
                    # Detection can still be displayed if IDs
                    # are temporarily unavailable.
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

                    position = get_position(
                        center_x,
                        frame_width
                    )

                    # Unique key for each tracked object.
                    # If no ID is available, use a temporary
                    # class key rather than crashing.
                    if track_id is not None:
                        track_key = f"id_{track_id}"
                    else:
                        track_key = f"untracked_{object_name}"

                    # Show the tracking ID and position.
                    if track_id is not None:
                        info_text = (
                            f"ID {track_id}: {object_name} "
                            f"| {position} | {confidence:.2f}"
                        )
                    else:
                        info_text = (
                            f"{object_name} | {position} "
                            f"| {confidence:.2f}"
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

                    # Only announce detections that have IDs.
                    # Untracked detections remain visible.
                    if track_id is None:
                        continue

                    announce, reason = should_announce(
                        track_key,
                        object_name,
                        position,
                        current_time
                    )

                    if announce:

                        message = create_message(
                            object_name,
                            position
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

                    # Update last-seen time for this individual.
                    last_seen[track_key] = current_time

            # Show camera output.
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
        
