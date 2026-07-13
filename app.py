from __future__ import annotations

import math
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from tkinter import Tk, StringVar, Text, Button, Label, filedialog, messagebox
from tkinter import ttk

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
MODEL_PATH = Path("models/pose_landmarker_lite.task")


class PoseIndex:
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


@dataclass
class FrameMetrics:
    timestamp_ms: int
    left_elbow_angle: float | None
    right_elbow_angle: float | None
    body_line_angle: float | None
    elbow_width_ratio: float | None
    shoulder_y: float | None
    hip_y: float | None


@dataclass
class AnalysisResult:
    duration_sec: float
    analyzed_frames: int
    detected_frames: int
    repetitions: int
    advice: list[str]


def ensure_model_exists() -> None:
    MODEL_PATH.parent.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        return
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def angle_degrees(a, b, c) -> float:
    ab = np.array([a.x - b.x, a.y - b.y])
    cb = np.array([c.x - b.x, c.y - b.y])
    denominator = np.linalg.norm(ab) * np.linalg.norm(cb)
    if denominator == 0:
        return 0.0
    cosine = float(np.dot(ab, cb) / denominator)
    cosine = max(-1.0, min(1.0, cosine))
    return math.degrees(math.acos(cosine))


def distance(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def midpoint(a, b):
    return type("Point", (), {
        "x": (a.x + b.x) / 2,
        "y": (a.y + b.y) / 2,
    })()


def landmarks_are_visible(landmarks, indexes: list[int], threshold: float = 0.35) -> bool:
    for index in indexes:
        landmark = landmarks[index]
        if getattr(landmark, "visibility", 1.0) < threshold:
            return False
    return True


def elbow_angle_if_visible(landmarks, shoulder_index: int, elbow_index: int, wrist_index: int) -> float | None:
    indexes = [shoulder_index, elbow_index, wrist_index]
    if not landmarks_are_visible(landmarks, indexes):
        return None
    return angle_degrees(*(landmarks[index] for index in indexes))


def metrics_from_landmarks(landmarks, timestamp_ms: int) -> FrameMetrics | None:
    left_elbow_angle = elbow_angle_if_visible(
        landmarks, PoseIndex.LEFT_SHOULDER, PoseIndex.LEFT_ELBOW, PoseIndex.LEFT_WRIST
    )
    right_elbow_angle = elbow_angle_if_visible(
        landmarks, PoseIndex.RIGHT_SHOULDER, PoseIndex.RIGHT_ELBOW, PoseIndex.RIGHT_WRIST
    )
    if left_elbow_angle is None and right_elbow_angle is None:
        return None

    torso_indexes = [
        PoseIndex.LEFT_SHOULDER,
        PoseIndex.RIGHT_SHOULDER,
        PoseIndex.LEFT_HIP,
        PoseIndex.RIGHT_HIP,
    ]
    body_line_indexes = torso_indexes + [
        PoseIndex.LEFT_ANKLE,
        PoseIndex.RIGHT_ANKLE,
    ]

    left_shoulder = landmarks[PoseIndex.LEFT_SHOULDER]
    right_shoulder = landmarks[PoseIndex.RIGHT_SHOULDER]
    shoulder_y = None
    hip_y = None
    body_line_angle = None
    if landmarks_are_visible(landmarks, torso_indexes):
        left_hip = landmarks[PoseIndex.LEFT_HIP]
        right_hip = landmarks[PoseIndex.RIGHT_HIP]
        shoulder_center = midpoint(left_shoulder, right_shoulder)
        hip_center = midpoint(left_hip, right_hip)
        shoulder_y = shoulder_center.y
        hip_y = hip_center.y
        if landmarks_are_visible(landmarks, body_line_indexes):
            ankle_center = midpoint(
                landmarks[PoseIndex.LEFT_ANKLE], landmarks[PoseIndex.RIGHT_ANKLE]
            )
            body_line_angle = angle_degrees(shoulder_center, hip_center, ankle_center)

    elbow_width_ratio = None
    elbow_indexes = [PoseIndex.LEFT_ELBOW, PoseIndex.RIGHT_ELBOW]
    shoulder_indexes = [PoseIndex.LEFT_SHOULDER, PoseIndex.RIGHT_SHOULDER]
    if landmarks_are_visible(landmarks, elbow_indexes + shoulder_indexes):
        shoulder_width = max(distance(left_shoulder, right_shoulder), 0.001)
        elbow_width_ratio = (
            distance(landmarks[PoseIndex.LEFT_ELBOW], landmarks[PoseIndex.RIGHT_ELBOW])
            / shoulder_width
        )

    return FrameMetrics(
        timestamp_ms=timestamp_ms,
        left_elbow_angle=left_elbow_angle,
        right_elbow_angle=right_elbow_angle,
        body_line_angle=body_line_angle,
        elbow_width_ratio=elbow_width_ratio,
        shoulder_y=shoulder_y,
        hip_y=hip_y,
    )


def count_repetitions(metrics: list[FrameMetrics]) -> int:
    """Count completed down-to-up cycles from the arm with the clearest motion."""
    left_angles = [frame.left_elbow_angle for frame in metrics if frame.left_elbow_angle is not None]
    right_angles = [frame.right_elbow_angle for frame in metrics if frame.right_elbow_angle is not None]
    return max(count_repetitions_from_angles(left_angles), count_repetitions_from_angles(right_angles))


def count_repetitions_from_angles(angles: list[float]) -> int:
    if len(angles) < 6:
        return 0

    smoothed = [
        mean(angles[max(0, index - 1): min(len(angles), index + 2)])
        for index in range(len(angles))
    ]

    # Derive both phase thresholds from this video's observed range. Percentile
    # thresholds can miss a single short bottom phase: in a four-second clip it
    # may occupy fewer than 20% of frames. A minimum range filters landmark
    # jitter while allowing one full, clearly visible repetition.
    min_angle = min(smoothed)
    max_angle = max(smoothed)
    movement_range = max_angle - min_angle
    if movement_range < 25.0:
        return 0
    low_threshold = min_angle + movement_range * 0.25
    high_threshold = max_angle - movement_range * 0.25

    state = "bottom" if smoothed[0] <= low_threshold else "top"
    reps = 0
    for angle in smoothed:
        if state == "top" and angle <= low_threshold:
            state = "bottom"
        elif state == "bottom" and angle >= high_threshold:
            reps += 1
            state = "top"
    return reps

def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percent)))
    return ordered[index]


MAX_RECOMMENDED_VIDEO_DURATION_SEC = 30


def build_pushup_advice(metrics: list[FrameMetrics], duration_sec: float) -> list[str]:
    advice: list[str] = []
    if duration_sec > MAX_RECOMMENDED_VIDEO_DURATION_SEC:
        advice.append(
            "Для более точного анализа используйте видео длительностью до 30 секунд, без лишних пауз."
        )

    if not metrics:
        return [
            "Не удалось уверенно найти позу. Снимите человека полностью в кадре, при хорошем освещении, без закрытия рук, корпуса и ног."
        ]

    elbow_angles = [
        angle
        for frame in metrics
        for angle in (frame.left_elbow_angle, frame.right_elbow_angle)
        if angle is not None
    ]
    min_elbow = min(elbow_angles)
    max_elbow = max(elbow_angles)
    body_line_angles = [frame.body_line_angle for frame in metrics if frame.body_line_angle is not None]
    elbow_width_ratios = [frame.elbow_width_ratio for frame in metrics if frame.elbow_width_ratio is not None]
    shoulder_positions = [frame.shoulder_y for frame in metrics if frame.shoulder_y is not None]
    hip_shoulder_gaps = [
        abs(frame.hip_y - frame.shoulder_y)
        for frame in metrics
        if frame.hip_y is not None and frame.shoulder_y is not None
    ]

    if elbow_width_ratios and percentile(elbow_width_ratios, 0.9) > 1.85:
        advice.append(
            "Локти выглядят слишком широко. Попробуйте держать кисти примерно под плечами и уводить локти назад под умеренным углом, а не строго в стороны."
        )

    if body_line_angles and mean(body_line_angles) < 165:
        advice.append(
            "Корпус часто теряет прямую линию. Держите плечи, таз и стопы на одной линии, без провала поясницы и без подъема таза вверх."
        )

    if min_elbow > 100:
        advice.append(
            "Нижняя фаза неполная: локти сгибаются недостаточно. Опускайтесь ниже, пока грудь не окажется близко к полу, сохраняя корпус прямым."
        )

    if max_elbow < 150:
        advice.append(
            "Верхняя фаза неполная: руки почти не выпрямляются. Поднимайтесь до почти прямых локтей, но без резкого переразгибания."
        )

    if shoulder_positions and max(shoulder_positions) - min(shoulder_positions) < 0.08:
        advice.append(
            "Амплитуда движения маленькая. Проверьте, что в видео есть полное опускание и подъем, а камера не стоит слишком близко."
        )

    if hip_shoulder_gaps and mean(hip_shoulder_gaps) > 0.18:
        advice.append(
            "Плечи и таз сильно расходятся по высоте. Снимайте сбоку и держите корпус жестким, чтобы таз не проваливался и не уходил вверх."
        )

    left_right_gaps = [
        abs(frame.left_elbow_angle - frame.right_elbow_angle)
        for frame in metrics
        if frame.left_elbow_angle is not None and frame.right_elbow_angle is not None
    ]
    if left_right_gaps and mean(left_right_gaps) > 22:
        advice.append(
            "Левая и правая рука работают несимметрично. Следите, чтобы плечи опускались и поднимались ровно, без перекоса корпуса."
        )

    if not advice:
        advice.append(
            "Критичных ошибок по текущим правилам не найдено. Сохраняйте полный диапазон движения и контролируемый темп."
        )

    return advice


def analyze_pushup_video(video_path: str) -> AnalysisResult:
    ensure_model_exists()

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError("Не удалось открыть видеофайл.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = frame_count / fps if frame_count else 0.0

    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )

    metrics: list[FrameMetrics] = []
    analyzed_frames = 0
    detected_frames = 0
    step = max(1, round(fps / 15))  # About 15 analyzed frames per second.

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % step != 0:
                frame_index += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_index * 1000 / fps)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            analyzed_frames += 1

            if result.pose_landmarks:
                detected_frames += 1
                frame_metrics = metrics_from_landmarks(
                    result.pose_landmarks[0], timestamp_ms
                )
                if frame_metrics:
                    metrics.append(frame_metrics)

            frame_index += 1

    capture.release()

    reps = count_repetitions(metrics)
    advice = build_pushup_advice(metrics, duration_sec)
    detection_ratio = detected_frames / analyzed_frames if analyzed_frames else 0
    if detection_ratio < 0.3:
        advice.insert(
            0,
            "Позу удаётся распознать только в части ролика, поэтому вывод ориентировочный. Для более точной оценки держите в кадре плечи, руки, таз и стопы.",
        )

    return AnalysisResult(
        duration_sec=duration_sec,
        analyzed_frames=analyzed_frames,
        detected_frames=detected_frames,
        repetitions=reps,
        advice=advice,
    )


class PushupAnalyzerApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("Анализ отжиманий")
        self.root.geometry("780x560")
        self.video_path = StringVar(value="Видео не выбрано")
        self.status = StringVar(value="Выберите видео длительностью до 30 секунд.")

        main = ttk.Frame(root, padding=16)
        main.pack(fill="both", expand=True)

        title = ttk.Label(main, text="Анализ техники отжиманий", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w")

        angle_text = (
            "Ракурс для первой версии: снимайте сбоку под 30-45 градусов, камера на уровне корпуса, "
            "человек полностью в кадре, обе руки и стопы видны."
        )
        ttk.Label(main, text=angle_text, wraplength=720).pack(anchor="w", pady=(8, 12))

        controls = ttk.Frame(main)
        controls.pack(fill="x")

        Button(controls, text="Выбрать видео", command=self.choose_video).pack(side="left")
        Button(controls, text="Анализировать", command=self.start_analysis).pack(side="left", padx=8)

        Label(main, textvariable=self.video_path, anchor="w").pack(fill="x", pady=(12, 4))
        Label(main, textvariable=self.status, anchor="w").pack(fill="x", pady=(0, 12))

        self.output = Text(main, wrap="word", height=20)
        self.output.pack(fill="both", expand=True)
        self.output.insert(
            "1.0",
            "Здесь появится результат анализа.\n\n"
            "Важно: это учебный прототип. Он дает подсказки по ключевым точкам тела, "
            "но не заменяет тренера и не учитывает медицинские ограничения.",
        )
        self.output.configure(state="disabled")

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                ("Видео", "*.mp4 *.mov *.avi *.mkv"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self.video_path.set(path)
            self.status.set("Видео выбрано. Нажмите 'Анализировать'.")

    def start_analysis(self) -> None:
        path = self.video_path.get()
        if not Path(path).exists():
            messagebox.showwarning("Нет видео", "Сначала выберите видеофайл.")
            return

        self.status.set("Идет анализ. При первом запуске модель MediaPipe будет скачана.")
        self.write_output("Анализирую видео, подождите...\n")
        thread = threading.Thread(target=self.run_analysis, args=(path,), daemon=True)
        thread.start()

    def run_analysis(self, path: str) -> None:
        try:
            result = analyze_pushup_video(path)
            lines = [
                "Результат анализа",
                "",
                f"Длительность видео: {result.duration_sec:.1f} сек.",
                f"Проанализировано кадров: {result.analyzed_frames}",
                f"Кадров с найденной позой: {result.detected_frames}",
                f"Примерное число повторений: {result.repetitions}",
                "",
                "Советы:",
            ]
            lines.extend(f"- {item}" for item in result.advice)
            text = "\n".join(lines)
            self.root.after(0, lambda: self.finish_success(text))
        except Exception as exc:
            self.root.after(0, lambda: self.finish_error(str(exc)))

    def finish_success(self, text: str) -> None:
        self.status.set("Анализ завершен.")
        self.write_output(text)

    def finish_error(self, error: str) -> None:
        self.status.set("Ошибка анализа.")
        self.write_output(f"Ошибка:\n{error}")

    def write_output(self, text: str) -> None:
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")


def main() -> None:
    root = Tk()
    PushupAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
