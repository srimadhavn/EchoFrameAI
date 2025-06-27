import argparse
import os
import cv2
from ultralytics import YOLO
from gtts import gTTS
from math import sqrt, atan2,degrees
import subprocess
import mediapipe as mp

model = YOLO("yolov8x.pt")

# --------------------- POSE DETECTION ---------------------
def calculate_angle(a, b, c):
    """Calculate angle at point b given three landmark points."""
    ang = degrees(
        atan2(c.y - b.y, c.x - b.x) - atan2(a.y - b.y, a.x - b.x)
    )
    ang = abs(ang)
    return ang if ang <= 180 else 360 - ang


def detect_pose(image_path):
    mp_pose = mp.solutions.pose
    pose_detector = mp_pose.Pose(static_image_mode=True)
    
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(image_rgb)

    if not results.pose_landmarks:
        return "unknown"

    landmarks = results.pose_landmarks.landmark

    l_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    l_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    r_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    l_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    r_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    avg_hip_y = (l_hip.y + r_hip.y) / 2
    avg_knee_y = (l_knee.y + r_knee.y) / 2
    vertical_dy = abs(avg_hip_y - avg_knee_y)

    hip_knee_dx = abs((l_hip.x + r_hip.x)/2 - (l_knee.x + r_knee.x)/2)

    left_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    avg_knee_angle = (left_knee_angle + right_knee_angle) / 2

    if avg_knee_angle > 160 and vertical_dy > 0.3 and hip_knee_dx < 0.1:
        return "standing"
    elif avg_knee_angle < 120 or hip_knee_dx > 0.15:
        return "sitting"
    else:
        return "unclear"
# --------------------- YOLO DETECTION ---------------------
def run_yolo(image_path, conf_threshold=0.6):
    results = model(image_path)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/annotated.jpg", results[0].plot())

    object_data = []
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue
            cls = int(box.cls[0])
            name = model.names[cls] if model.names is not None else f"class_{cls}"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center = [(x1 + x2)/2, (y1 + y2)/2]
            object_data.append({
                "name": name,
                "confidence": round(conf, 2),
                "bbox": [round(x1), round(y1), round(x2), round(y2)],
                "center": [round(center[0]), round(center[1])]
            })
    return object_data

# --------------------- RELATIONSHIP DETECTION ---------------------
def detect_relationships(objects, threshold=1000):
    relationships = []
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            o1, o2 = objects[i], objects[j]
            dx = o1["center"][0] - o2["center"][0]
            dy = o1["center"][1] - o2["center"][1]
            if sqrt(dx*dx + dy*dy) < threshold:
                relationships.append(f"{o1['name']} is near {o2['name']}")
    return relationships

# --------------------- PROMPT GENERATION ---------------------
def build_prompt(objects, interactions, pose_label):
    summary = {}
    for obj in objects:
        summary[obj["name"]] = summary.get(obj["name"], 0) + 1
    obj_lines = [f"- {v} {k + 's'*(v>1)}" for k, v in summary.items()]
    rel_lines = [f"- {r}" for r in interactions]

    return (
        "You are an assistant that tells a story based on a detected scene.\n"
        f"Detected Pose: {pose_label}\n\n"
        "Detected Objects:\n" + "\n".join(obj_lines) + "\n\n"
        "Detected Relationships:\n" + "\n".join(rel_lines) + "\n\n"
        "Now tell the story:\n"
    )

# --------------------- LLaMA + AUDIO ---------------------
def run_llama(prompt):
    result = subprocess.run(["ollama", "run", "llama3.2"],
                            input=prompt.encode(),
                            stdout=subprocess.PIPE)
    return result.stdout.decode().strip()

def save_and_speak(story_text):
    with open("output/story.txt", "w") as f:
        f.write(story_text)
    gTTS(story_text, lang="en").save("output/story.mp3")
    print("🎧 Audio saved to output/story.mp3")

# --------------------- MAIN ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    args = parser.parse_args()

    print("🔍 Detecting objects...")
    objects = run_yolo(args.image)

    print("📐 Finding relationships...")
    relations = detect_relationships(objects)

    print("🧍 Detecting pose...")
    pose = detect_pose(args.image)
    print(f" Pose: {pose}")

    print(" Prompting LLaMA...")
    prompt = build_prompt(objects, relations, pose)

    print(" Generating story...")
    story = run_llama(prompt)
    print(" Story:\n", story)

    print(" Generating audio...")
    save_and_speak(story)

if __name__ == "__main__":
    main()