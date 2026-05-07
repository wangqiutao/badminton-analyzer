import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import tempfile
import time
import os
from scipy.spatial.distance import euclidean
import urllib.request  # 确保在顶部导入，防止模型下载失败

# ==================== 配置与数据结构 ====================
@dataclass
class AnalysisConfig:
    """分析参数配置"""
    model_complexity: int = 2
    min_detection_conf: float = 0.5
    min_tracking_conf: float = 0.5
    history_length: int = 45
    hit_cooldown: int = 15
    speed_threshold: float = 30.0
    hit_confidence: float = 0.65
    ideal_angles = {
        'forehand_clear': {'elbow': 160, 'shoulder': 110, 'knee': 140, 'trunk_tilt': 15},
        'smash':          {'elbow': 175, 'shoulder': 135, 'knee': 125, 'trunk_tilt': 25},
        'drop_shot':      {'elbow': 145, 'shoulder': 95,  'knee': 150, 'trunk_tilt': 5},
        'net_kill':       {'elbow': 155, 'shoulder': 105, 'knee': 145, 'trunk_tilt': 20},
    }
    joint_weights = {'elbow': 0.3, 'shoulder': 0.3, 'knee': 0.2, 'trunk_tilt': 0.2}

# ==================== 姿态估计器 ====================
class PoseEstimator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        model_path = self._download_model()
        options = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.min_detection_conf,
            min_pose_presence_confidence=config.min_tracking_conf,
            min_tracking_confidence=config.min_tracking_conf,
            output_segmentation_masks=False
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)
        self.timestamp = 0

    def _download_model(self) -> str:
        model_dir = os.path.join(os.path.expanduser('~'), '.mediapipe_models')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'pose_landmarker_heavy.task')
        if not os.path.exists(model_path):
            st.info("首次运行，正在下载姿态模型 (约 12MB)...")
            url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task'
            urllib.request.urlretrieve(url, model_path)
        return model_path

    def process_frame(self, frame: np.ndarray):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.timestamp += 33
        detection_result = self.landmarker.detect_for_video(mp_image, self.timestamp)
        if not detection_result.pose_landmarks:
            return None
        h, w = frame.shape[:2]
        landmarks = detection_result.pose_landmarks[0]
        points = {
            'left_shoulder': self._get_pixel(landmarks, 11, w, h),
            'right_shoulder': self._get_pixel(landmarks, 12, w, h),
            'left_elbow': self._get_pixel(landmarks, 13, w, h),
            'right_elbow': self._get_pixel(landmarks, 14, w, h),
            'left_wrist': self._get_pixel(landmarks, 15, w, h),
            'right_wrist': self._get_pixel(landmarks, 16, w, h),
            'left_hip': self._get_pixel(landmarks, 23, w, h),
            'right_hip': self._get_pixel(landmarks, 24, w, h),
            'left_knee': self._get_pixel(landmarks, 25, w, h),
            'right_knee': self._get_pixel(landmarks, 26, w, h),
            'left_ankle': self._get_pixel(landmarks, 27, w, h),
            'right_ankle': self._get_pixel(landmarks, 28, w, h),
        }
        if points['right_wrist']['y'] < points['left_wrist']['y']:
            dominant = 'right'
        else:
            dominant = 'left'
        return {'points': points, 'dominant': dominant}

    def _get_pixel(self, landmarks, idx, w, h):
        lm = landmarks[idx]
        return {'x': lm.x * w, 'y': lm.y * h, 'z': lm.z, 'vis': lm.visibility}

# ==================== 角度计算与特征提取 ====================
def calculate_angle(p1, p2, p3):
    a = np.array([p1['x'], p1['y']])
    b = np.array([p2['x'], p2['y']])
    c = np.array([p3['x'], p3['y']])
    ba = a - b
    bc = c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine, -1.0, 1.0))
    return np.degrees(angle)

def extract_features(pose_data):
    pts = pose_data['points']
    side = pose_data['dominant']
    s = side
    features = {
        'elbow': calculate_angle(pts[f'{s}_shoulder'], pts[f'{s}_elbow'], pts[f'{s}_wrist']),
        'shoulder': calculate_angle(pts[f'{s}_hip'], pts[f'{s}_shoulder'], pts[f'{s}_elbow']),
        'knee': calculate_angle(pts[f'{s}_hip'], pts[f'{s}_knee'], pts[f'{s}_ankle']),
    }
    shoulder_mid = np.array([pts['left_shoulder']['x'] + pts['right_shoulder']['x'],
                             pts['left_shoulder']['y'] + pts['right_shoulder']['y']]) / 2
    hip_mid = np.array([pts['left_hip']['x'] + pts['right_hip']['x'],
                        pts['left_hip']['y'] + pts['right_hip']['y']]) / 2
    dx = shoulder_mid[0] - hip_mid[0]
    dy = shoulder_mid[1] - hip_mid[1]
    features['trunk_tilt'] = np.degrees(np.arctan2(dx, -dy))
    wrist = pts[f'{s}_wrist']
    elbow = pts[f'{s}_elbow']
    features['wrist'] = np.degrees(np.arctan2(wrist['y'] - elbow['y'], wrist['x'] - elbow['x']))
    wrist_pos = (pts[f'{s}_wrist']['x'], pts[f'{s}_wrist']['y'])
    return features, wrist_pos

# ==================== 动作分类器 ====================
class TransformerClassifier:
    def __init__(self):
        self.model = None
        self.label_map = {0: 'smash', 1: 'forehand_clear', 2: 'drop_shot', 3: 'net_kill'}

    def predict(self, sequence_features):
        return None, 0.0

class ActionDetector:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.feature_history = deque(maxlen=config.history_length)
        self.wrist_history = deque(maxlen=config.history_length)
        self.last_hit_frame = -config.hit_cooldown
        self.frame_idx = 0
        self.transformer = TransformerClassifier()

    def update(self, features, wrist_pos):
        self.frame_idx += 1
        self.feature_history.append(features)
        self.wrist_history.append(wrist_pos)
        if len(self.feature_history) < 10:
            return False, 'none', 0.0
        if self.frame_idx - self.last_hit_frame < self.config.hit_cooldown:
            return False, 'none', 0.0
        recent_wrist = list(self.wrist_history)[-5:]
        speeds = [euclidean(recent_wrist[i], recent_wrist[i-1]) for i in range(1, len(recent_wrist))]
        max_speed = max(speeds) if speeds else 0
        if max_speed < self.config.speed_threshold:
            return False, 'none', 0.0
        action, conf = self.transformer.predict(np.array(self.feature_history))
        if action is not None and conf > self.config.hit_confidence:
            self.last_hit_frame = self.frame_idx
            return True, action, conf
        action, conf = self._rule_based_classify()
        if conf > self.config.hit_confidence:
            self.last_hit_frame = self.frame_idx
            return True, action, conf
        return False, 'none', 0.0

    def _rule_based_classify(self):
        recent = list(self.feature_history)[-15:]
        avg = {k: np.mean([f[k] for f in recent]) for k in recent[0].keys()}
        scores = {
            'smash':  (avg['shoulder'] / 135) * 0.3 + (avg['trunk_tilt'] / 25) * 0.3 + (avg['elbow'] / 175) * 0.4,
            'forehand_clear': (avg['shoulder'] / 110) * 0.4 + (avg['elbow'] / 160) * 0.4 + (avg['knee'] / 140) * 0.2,
            'drop_shot': (95 / max(avg['shoulder'], 1)) * 0.3 + (145 / max(avg['elbow'], 1)) * 0.3 + (avg['knee'] / 150) * 0.4,
            'net_kill': (avg['shoulder'] / 105) * 0.3 + (avg['elbow'] / 155) * 0.3 + (avg['trunk_tilt'] / 20) * 0.4,
        }
        best_action = max(scores, key=scores.get)
        conf = min(scores[best_action], 1.0)
        return best_action, conf

# ==================== 动作评估器 ====================
class SwingEvaluator:
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.records = []

    def evaluate(self, angles, action_type):
        ideal = self.config.ideal_angles.get(action_type)
        if not ideal:
            return {'total': 0, 'details': {}, 'grade': '未知', 'feedback': []}
        weights = self.config.joint_weights
        total_score = 0
        total_weight = 0
        details = {}
        feedback = []
        for joint, ideal_val in ideal.items():
            if joint in angles:
                actual = angles[joint]
                diff = abs(actual - ideal_val)
                score = max(0, 100 - diff * 0.8)
                details[joint] = {'actual': round(actual, 1), 'ideal': ideal_val, 'score': round(score, 1)}
                w = weights.get(joint, 0)
                total_score += score * w
                total_weight += w
                if diff > 25:
                    hint = "偏大" if actual > ideal_val else "偏小"
                    feedback.append(f"{joint} 角度{hint} (差距 {diff:.0f}°)")
        final = round(total_score / total_weight, 1) if total_weight > 0 else 0
        if final >= 90: grade = "优秀"
        elif final >= 80: grade = "良好"
        elif final >= 70: grade = "中等"
        elif final >= 60: grade = "及格"
        else: grade = "需改进"
        result = {'total': final, 'grade': grade, 'details': details, 'feedback': feedback}
        self.records.append({'action': action_type, 'score': final, 'angles': angles})
        return result

# ==================== 可视化工具 ====================
def draw_pose_and_trajectory(frame, pose_data, wrist_trail, action_text=""):
    vis = frame.copy()
    if pose_data:
        pts = pose_data['points']
        connections = [
            ('left_shoulder', 'right_shoulder'), ('left_shoulder', 'left_elbow'),
            ('left_elbow', 'left_wrist'), ('right_shoulder', 'right_elbow'),
            ('right_elbow', 'right_wrist'), ('left_shoulder', 'left_hip'),
            ('right_shoulder', 'right_hip'), ('left_hip', 'right_hip'),
            ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
            ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
        ]
        for c in connections:
            p1 = pts[c[0]]
            p2 = pts[c[1]]
            if p1['vis'] > 0.5 and p2['vis'] > 0.5:
                cv2.line(vis, (int(p1['x']), int(p1['y'])), (int(p2['x']), int(p2['y'])), (0,255,0), 2)
    for i in range(1, len(wrist_trail)):
        cv2.line(vis, tuple(wrist_trail[i-1]), tuple(wrist_trail[i]), (0,200,200), 2)
    if action_text:
        cv2.putText(vis, action_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return vis

# ==================== 主分析器 ====================
def analyze_video(video_path, config, progress_callback):
    pose_estimator = PoseEstimator(config)
    action_detector = ActionDetector(config)
    evaluator = SwingEvaluator(config)
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    swings = []
    wrist_trail = deque(maxlen=100)
    frame_idx = 0
    last_vis = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        if progress_callback:
            progress_callback(frame_idx / total_frames)
        pose_data = pose_estimator.process_frame(frame)
        if not pose_data:
            continue
        angles, wrist_pos = extract_features(pose_data)
        wrist_trail.append((int(wrist_pos[0]), int(wrist_pos[1])))
        is_hit, action_type, confidence = action_detector.update(angles, wrist_pos)
        if is_hit:
            eval_result = evaluator.evaluate(angles, action_type)
            swings.append({
                'frame': frame_idx,
                'time': round(frame_idx / fps, 2),
                'action': action_type,
                'confidence': round(confidence, 2),
                'evaluation': eval_result,
                'angles': angles
            })
        if frame_idx % 15 == 0 or is_hit:
            last_vis = draw_pose_and_trajectory(frame, pose_data, list(wrist_trail),
                                                f"{action_type} ({confidence:.2f})" if is_hit else "")
    cap.release()
    return {
        'swings': swings,
        'total_frames': frame_idx,
        'fps': fps,
        'statistics': _compute_statistics(swings, evaluator),
        'last_frame': last_vis
    }

def _compute_statistics(swings, evaluator):
    if not swings:
        return {}
    scores = [s['evaluation']['total'] for s in swings]
    action_counts = {}
    for s in swings:
        act = s['action']
        action_counts[act] = action_counts.get(act, 0) + 1
    return {
        'total_swings': len(swings),
        'avg_score': round(np.mean(scores), 1),
        'max_score': max(scores),
        'min_score': min(scores),
        'action_distribution': action_counts,
        'records': evaluator.records
    }

# ==================== Streamlit 界面 ====================
st.set_page_config(page_title="羽毛球动作分析 Pro", layout="wide")
st.title("🏸 羽毛球智能动作分析系统 (手机版)")

with st.sidebar:
    st.header("⚙️ 参数配置")
    complexity = st.selectbox("模型复杂度", [0,1,2], index=2, format_func=lambda x:["轻量","标准","精准"][x])
    hit_threshold = st.slider("击球检测灵敏度", 0.3, 1.0, 0.65, 0.05)
    st.markdown("---")
    st.info("上传侧面拍摄的挥拍视频，光线充足，距离3-5米")

uploaded_file = st.file_uploader("选择视频文件", type=["mp4","avi","mov","mkv"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name
    st.video(video_path)
    if st.button("开始分析", type="primary"):
        config = AnalysisConfig(model_complexity=complexity, hit_confidence=hit_threshold)
        progress_bar = st.progress(0)
        status = st.empty()
        status.text("分析中，请稍候...")
        start_time = time.time()
        results = analyze_video(video_path, config,
                                lambda p: progress_bar.progress(min(p, 1.0)))
        elapsed = time.time() - start_time
        status.success(f"✅ 分析完成！耗时 {elapsed:.1f} 秒")
        progress_bar.empty()
        st.markdown("---")
        st.header("📊 分析报告")
        stats = results['statistics']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("总挥拍次数", stats.get('total_swings', 0))
        col2.metric("平均评分", f"{stats.get('avg_score', 0)} 分")
        col3.metric("最高评分", stats.get('max_score', 0))
        col4.metric("最低评分", stats.get('min_score', 0))
        st.subheader("🎯 动作分布")
        action_dist = stats.get('action_distribution', {})
        if action_dist:
            st.bar_chart(pd.DataFrame({'动作': list(action_dist.keys()), '次数': list(action_dist.values())}).set_index('动作'))
        swings = results['swings']
        if swings:
            st.subheader("📋 详细挥拍记录")
            for i, swing in enumerate(swings, 1):
                ev = swing['evaluation']
                with st.expander(f"第{i}次 | {swing['action']} | 评分 {ev['total']} | {ev['grade']} | 置信度 {swing['confidence']}"):
                    c1, c2 = st.columns(2)
                    with c1:
                        for joint, info in ev['details'].items():
                            st.write(f"• {joint}: {info['actual']}° (理想 {info['ideal']}°, 得分 {info['score']})")
                    with c2:
                        if ev['feedback']:
                            for fb in ev['feedback']:
                                st.warning(fb)
                        else:
                            st.success("动作标准！")
            st.subheader("📈 动作标准度趋势")
            scores_list = [s['evaluation']['total'] for s in swings]
            fig, ax = plt.subplots()
            ax.plot(range(1, len(scores_list)+1), scores_list, 'o-')
            ax.axhline(90, color='g', linestyle='--', label='优秀')
            ax.axhline(70, color='r', linestyle='--', label='及格')
            ax.set_xlabel('挥拍序号')
            ax.set_ylabel('评分')
            ax.legend()
            st.pyplot(fig)
        if results.get('last_frame') is not None:
            st.subheader("🎥 姿态与轨迹示例")
            st.image(results['last_frame'], channels="BGR", use_column_width=True)
    os.unlink(video_path)