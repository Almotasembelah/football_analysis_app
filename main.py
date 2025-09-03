import streamlit as st
from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import time 
import tempfile
import os
from pathlib import Path
from typing import Optional, Tuple, List, Any
import cv2
import numpy as np
import logging

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from football_utils import draw_annotation, draw_ball_annotation, draw_farthest_players, draw_keypoints, ImprovedPlayers, Transform, load_video, show_video
from assigners import Player2TeamAssigner
from matplotlib.colors import Normalize
import matplotlib
from huggingface_hub import hf_hub_download

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_file_from_hf(repo, filename, local_path):
    if not os.path.exists(local_path):
        print(f"Downloading {filename} from Hugging Face...")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        downloaded_path = hf_hub_download(repo_id=repo, filename=filename)
        os.rename(downloaded_path, local_path)
    return local_path

# Configuration constants
class Config:
    MODEL_PATH = get_file_from_hf("Almotasembelah/football-analysis-assets", "/models/players/best.pt", 'models/players/best.pt')
    BALL_MODEL_PATH = get_file_from_hf("Almotasembelah/football-analysis-assets", "/models/ball_model/best.pt", 'models/ball_model/best.pt')
    KPT_MODEL_PATH = get_file_from_hf("Almotasembelah/football-analysis-assets", "/models/keypointsbest.pt", 'models/keypoints/best.pt')
    OUTPUT_VIDEO_PATH = "out/exp.mp4"
    
    # SAHI parameters
    SLICE_HEIGHT = 640
    SLICE_WIDTH = 640
    OVERLAP_HEIGHT_RATIO = 0.05
    OVERLAP_WIDTH_RATIO = 0.05
    CONFIDENCE_THRESHOLD = 0.4
    IOU_THRESHOLD = 0.1
    
    # YOLO parameters
    PLAYER_CONF_THRESHOLD = 0.1
    PLAYER_IOU_THRESHOLD = 0.6
    KEYPOINT_CONF_THRESHOLD = 0.3
    # BATCH_SIZE = 500

class FootballAnalyzer:
    """Main class for football video analysis."""
    
    def __init__(self,  players_det, kpts_det, offside_det, ann_kpts, field_lines, ball_ann):
        self.players_det, self.kpts_det, self.offside_det, \
            self.ann_kpts, self.field_lines, self.ball_ann =  players_det, kpts_det, offside_det, ann_kpts, field_lines, ball_ann
        self.yolo_model = None
        self.ball_model = None
        self.kpt_model = None
        self.team_assigner = Player2TeamAssigner()
        self.transform = Transform() if self.kpts_det else None
        self.players = ImprovedPlayers(self.transform)
        
    def load_models(self) -> None:
        """Load required models based on selected features."""
        try:
            if self.players_det:
                if not Path(Config.MODEL_PATH).exists():
                    raise FileNotFoundError(f"Player detection model not found: {Config.MODEL_PATH}")
                self.yolo_model = YOLO(Config.MODEL_PATH)
                logger.info("Player detection model loaded")
                
            if self.kpts_det:
                if not Path(Config.KPT_MODEL_PATH).exists():
                    raise FileNotFoundError(f"Keypoint model not found: {Config.KPT_MODEL_PATH}")
                self.kpt_model = YOLO(Config.KPT_MODEL_PATH)
                logger.info("Keypoint model loaded")

            if self.ball_ann:
                if not Path(Config.BALL_MODEL_PATH).exists():
                    raise FileNotFoundError(f"Ball model not found: {Config.BALL_MODEL_PATH}")
                self.ball_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov11",
                    model_path=Config.BALL_MODEL_PATH,
                    confidence_threshold=Config.CONFIDENCE_THRESHOLD,
                    device="cuda" if st.session_state.get('use_gpu', True) else "cpu"
                )
                logger.info("Ball detection model loaded")
                
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            st.error(f"Failed to load models: {str(e)}")
            raise
    
    def process_frame(self, frame) -> Tuple[Any, List, Any, Any]:
        """Process a single frame with selected detection methods."""
        results = None
        ball_results = []
        kpts_results = None
        
        try:
            # Player detection
            if self.players_det and self.yolo_model:
                results = self.yolo_model.track(
                    source=frame,
                    tracker="btrack.yaml",
                    conf=Config.PLAYER_CONF_THRESHOLD,
                    iou=Config.PLAYER_IOU_THRESHOLD,
                    # batch=Config.BATCH_SIZE,
                    persist=True
                )
            
            # Keypoint detection
            if self.kpts_det and self.kpt_model:
                kpts_results = self.kpt_model.predict(
                    frame,
                    conf=Config.KEYPOINT_CONF_THRESHOLD
                )
                self.transform.find_homography(kpts_results[0].keypoints.xy.squeeze(0))
                self.players.transform = self.transform

            # Ball detection for offside
            if self.ball_ann and self.ball_model:
                ball_results = [
                    get_sliced_prediction(
                        f,
                        self.ball_model,
                        slice_height=Config.SLICE_HEIGHT,
                        slice_width=Config.SLICE_WIDTH,
                        overlap_height_ratio=Config.OVERLAP_HEIGHT_RATIO,
                        overlap_width_ratio=Config.OVERLAP_WIDTH_RATIO,
                        postprocess_match_threshold=Config.IOU_THRESHOLD,
                        postprocess_type="NMS"
                    ).object_prediction_list 
                    for f in frame
                ]
                
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            st.error(f"Frame processing error: {str(e)}")
            
        return results, ball_results, kpts_results, frame

def setup_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="Football Analysis",
        page_icon="⚽",
        layout="wide"
    )

def create_sidebar():
    """Create sidebar with settings."""
    with st.sidebar:
        st.header("Settings")
        use_gpu = st.checkbox("Use GPU (if available)", value=True)
        st.session_state['use_gpu'] = use_gpu
        
        st.header("Processing Options")
        max_frames = st.number_input("Max frames to process (0 = all)", min_value=0, value=0)
        st.session_state['max_frames'] = max_frames
        
        frame_skip = st.number_input("Process every N frames", min_value=1, value=1)
        st.session_state['frame_skip'] = frame_skip

def get_video_source() -> Optional[str]:
    """Handle video source selection and upload."""
    video_source = st.radio(
        "Choose video source:",
        ("Upload your own", "Use sample video")
    )
    
    video_path = None
    
    if video_source == "Upload your own":
        uploaded_file = st.file_uploader(
            "Upload a video", 
            type=["mp4", "avi", "mov", "mkv"],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        if uploaded_file is not None:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            # Display video info
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                st.info(f"📹 Video Info: {width}x{height}, {fps:.1f} FPS, {duration:.1f}s ({frame_count} frames)")
                cap.release()
    
    elif video_source == "Use sample video":
        sample_path = get_file_from_hf("Almotasembelah/football-analysis-assets", "inputs/input.mp4", 'inputs/test.mp4')
        if Path(sample_path).exists():
            video_path = sample_path
        else:
            st.error(f"Sample video not found: {sample_path}")
    
    return video_path

def process_video(analyzer: FootballAnalyzer, video_path: str):
    """Process video with selected analysis options."""
    
    # Create output directory
    os.makedirs("out", exist_ok=True)
    
    # Initialize progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    img_container = st.empty()
    
    # Create stop button
    stop_button = st.button('⏹️ Stop Processing', key='stop_btn', type="secondary")
    
    try:
        # Load video
        status_text.text("Loading video...")
        load, save_frames, cap, fps = load_video(video_path, Config.OUTPUT_VIDEO_PATH)
        gframes = load()
        
        # Load models
        status_text.text("Loading AI models...")
        analyzer.load_models()
        
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        max_frames = st.session_state.get('max_frames', 0)
        frame_skip = st.session_state.get('frame_skip', 1)
        
        if max_frames > 0:
            total_frames = min(total_frames, max_frames)
        
        start_time = time.time()
        
        # Process frames
        for frames in gframes:
            if stop_button:
                st.warning("Processing stopped by user")
                break
                
            # Process batch of frames
            results, ball_results, kpts_results, frames = analyzer.process_frame(frames)
            
            for i, frame in enumerate(frames):
                if frame_count >= total_frames:
                    break
                    
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                try:
                    # Apply selected annotations
                    if analyzer.players_det:
                        if results and len(results) > i:
                            result = analyzer.players.fit(frame, results[i])
                            result = analyzer.team_assigner.fit_predict(frame, result, i=3)
                            if result:
                                frame = draw_annotation(frame, result, 
                                                        [analyzer.team_assigner.team1_color, analyzer.team_assigner.team2_color], 
                                                        annotate_ball=False)
                            
                    # annotate the detected ball
                    if analyzer.ball_ann and ball_results and len(ball_results) > i:
                        frame = draw_ball_annotation(frame, ball_results[i])
                    
                    # annotate field keypoints
                    if analyzer.ann_kpts and kpts_results and len(kpts_results) > i:
                        frame = draw_keypoints(frame, kpts_results[i])
                    
                    # draw tactical map
                    if analyzer.kpts_det and kpts_results and len(kpts_results) > i and analyzer.players_det:
                        frame = analyzer.transform.draw_tactical_board(frame, result, analyzer.team_assigner.team1_color, analyzer.team_assigner.team2_color)
                    
                    if analyzer.offside_det and 'result' in locals():
                        frame = draw_farthest_players(frame, result, 
                                                    draw_field_lines=analyzer.field_lines, 
                                                    draw_offside_lines=False)
                    # Update display
                    img_container.image(frame)
                    
                    # Calculate and display FPS
                    current_time = time.time()
                    elapsed_time = current_time - start_time
                    processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    # Update progress
                    progress = min(frame_count / total_frames, 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {frame_count}/{total_frames} frames "
                                    f"({processing_fps:.1f} FPS)")
                    
                    # Save frame
                    writer = save_frames(frame, fps)
                        
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {str(e)}")
                    continue
                
                frame_count += 1
        
        # Display player snapshots
        if analyzer.players.snapshots:
            st.markdown("### Players' Snapshots")
            display_player_snapshots(analyzer.players)
            st.markdown("### Players' Heatmaps")
            display_player_heatmaps(analyzer)

        # Cleanup
        if 'writer' in locals():
            writer.release()
        cap.release()
        
        # Final status
        total_time = time.time() - start_time
        status_text.text(f"✅ Complete! Processed {frame_count} frames in {total_time:.1f}s")
        progress_bar.progress(1.0)
        
        return True
            
    except Exception as e:
        logger.error(f"Error during video processing: {str(e)}")
        st.error(f"Processing failed: {str(e)}")
        return False

def display_player_snapshots(players: ImprovedPlayers):
    """Display player snapshots in a grid."""
    if not players.snapshots:
        return
        
    cols_per_row = 5
    num_players = len(players.snapshots)
    num_rows = (num_players + cols_per_row - 1) // cols_per_row
    
    for row in range(num_rows):
        columns = st.columns(cols_per_row)
        for col in range(cols_per_row):
            idx = row * cols_per_row + col
            if idx < num_players:
                img = players.snapshots[idx]
                player_id = list(players.ids)[idx] if idx < len(players.ids) else idx
                columns[col].image(img)
                columns[col].caption(f"Player {idx+1}")

def overlay_heatmap_on_board(heatmap: np.ndarray, board_path: str = "./input/pitch.jpg", alpha: float = 0.5) -> np.ndarray:
    """
    Overlay a heatmap on top of a tactical board image.

    Args:
        heatmap (np.ndarray): 2D array of heatmap values.
        board_path (str): Path to tactical board image.
        alpha (float): Transparency of the heatmap (0 = invisible, 1 = fully opaque).

    Returns:
        np.ndarray: RGB image with heatmap overlay.
    """
    # Load tactical board
    board = cv2.imread(board_path)
    board = cv2.resize(board, (600, 300))
    board = cv2.cvtColor(board, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Resize heatmap to match board size
    heatmap_resized = cv2.resize(heatmap, (board.shape[1], board.shape[0]))

    # Convert heatmap to color
    norm = Normalize(vmin=0, vmax=heatmap_resized.max() if heatmap_resized.max() > 0 else 1)
    colormap = matplotlib.colormaps.get_cmap("hot")
    heatmap_img = colormap(norm(heatmap_resized))[:, :, :3]  # RGB, no alpha
    heatmap_img = (heatmap_img * 255).astype(np.uint8)

    # Overlay heatmap on board
    overlay = cv2.addWeighted(heatmap_img, alpha, board, 1 - alpha, 0)
    return overlay

def display_player_heatmaps(analyzer:FootballAnalyzer):
    """Display players' heatmaps in Streamlit grid layout."""
    players = analyzer.players
    if not players.snapshots:
        st.write("No heatmaps to display.")
        return
        
    cols_per_row = 5
    num_players = len(players.snapshots)
    num_rows = (num_players + cols_per_row - 1) // cols_per_row

    player_ids = list(players.ids)  # stable ordering

    if analyzer.kpts_det and analyzer.players_det:
        for row in range(num_rows):
            columns = st.columns(cols_per_row)
            for col in range(cols_per_row):
                idx = row * cols_per_row + col
                if idx < num_players:
                    player_id = player_ids[idx]
                    heatmap = players.get_heatmap(player_id)

                    if heatmap is None:
                        continue
                    #heatmap = analyzer.transform.transform(heatmap)
                    heatmap_img = overlay_heatmap_on_board(heatmap, alpha=0.6)

                    columns[col].image(heatmap_img, use_container_width=True)
                    columns[col].caption(f"Player {int(player_id)}")

def main():
    """Main application function."""
    setup_page()
    
    st.title("⚽ Football Analysis App")
    st.markdown("Advanced AI-powered football video analysis with player detection, keypoints, and offside detection.")
    
    # Create sidebar
    create_sidebar()
    
    # Feature selection
    st.markdown("### ⚽ Analysis Options")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        players_det = st.checkbox(
            '👥 Players Detection',
            value=True,
            help="Detect and track players throughout the video."
        )

    with col2:
        kpts_det = st.checkbox(
            '🎯 Keypoints Detection',
            help="Detect field keypoints to enable tactical map overlays and heatmaps per player. "
                "If combined with Players Detection, tactical insights and heatmaps will be displayed. ",
            key='kpts_det'
        )

    with col3:
        offside_det = st.checkbox(
            '🚩 Check Offside',
            help="Identify players who *may* be in an offside position based on their location relative to the last defender. "
                "⚠️ Full offside logic (e.g., ball position, active play) will be implemented in future releases.",
            disabled=not players_det
        )

        s = st.radio('Based On', ['Hough Transform', 'Field Keypoints'], 
                     help="Current release supports preliminary detection using either field keypoints or Hough Transform. "
                          "If Hough Transform is selected, you can choose whether to visualize the detected lines. ",
                    disabled=not offside_det)
    with col4:
        col4.write('Annotation Options')
        ann_kpts = st.checkbox('Keypoints Annotaion',
                               value=True if kpts_det else False,
                               help='enables annotation of keypoints directly on the video frames.',
                               disabled=False if kpts_det else True)
        field_lines = st.checkbox('Draw detected Field Lines', help='Only works if **Players Detection** is enabeld',disabled=not (s == 'Hough Transform' and offside_det and players_det))
        ball_ann = st.checkbox('Detect the Ball with SAHI', help='This Option will slow down the Processing')


    # Video source selection
    st.markdown("### Video Input")
    video_path = get_video_source()
    
    # Processing section
    if video_path:
        st.markdown("### Processing")
        
        if st.button("🚀 Start Analysis", type="primary"):
            if not any([players_det, kpts_det, offside_det]):
                st.warning("Please select at least one analysis option.")
                return
            
            # Initialize analyzer
            analyzer = FootballAnalyzer(players_det, kpts_det, offside_det, ann_kpts, field_lines, ball_ann)
            
            # Process video
            start_time = time.time()
            success = process_video(analyzer, video_path)
            
            if success:
                processing_time = time.time() - start_time
                st.success(f'⏱️ Processing completed in {processing_time:.2f} seconds')
                
                # Display processed video
                st.markdown('### 🎬 Processed Video')
                if Path(Config.OUTPUT_VIDEO_PATH).exists():
                    show_video(Config.OUTPUT_VIDEO_PATH)
                else:
                    st.error("Output video not found")
    else:
        st.info("👆 Please select a video source to begin analysis")

if __name__ == "__main__":
    main()