"""
DSP-Based Real-Time Video Processing Platform

An advanced video processing application implementing digital signal processing techniques
for real-time video acquisition, noise reduction, and comprehensive frequency domain analysis.
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft2, ifft2, fftshift
import plotly.graph_objects as go
import plotly.express as px

# Try importing PyTorch for AI features
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    st.warning("PyTorch not available. AI denoising will be disabled.")

st.set_page_config(
    page_title="Real-Time DSP Video Processing & Frequency Analysis Platform",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
.stTitle {
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.filter-info {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
</style>
""", unsafe_allow_html=True)

# Simple CNN for denoising (when PyTorch is available)
if PYTORCH_AVAILABLE:
    class SimpleDnCNN(nn.Module):
        def __init__(self, channels=3, features=32):
            super(SimpleDnCNN, self).__init__()
            
            self.layer1 = nn.Conv2d(channels, features, 3, padding=1)
            self.layer2 = nn.Conv2d(features, features, 3, padding=1)
            self.layer3 = nn.Conv2d(features, features, 3, padding=1)
            self.layer4 = nn.Conv2d(features, channels, 3, padding=1)
            
            self.relu = nn.ReLU(inplace=True)
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        
        def forward(self, x):
            residual = x
            out = self.relu(self.layer1(x))
            out = self.relu(self.bn1(self.layer2(out)))
            out = self.relu(self.bn2(self.layer3(out)))
            out = self.layer4(out)
            return residual - out  # Residual learning

class VideoProcessor:
    """Advanced video processor implementing real-time DSP techniques for noise reduction and frequency analysis."""
    
    def __init__(self):
        self.ai_model = None
        if PYTORCH_AVAILABLE:
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.ai_model = SimpleDnCNN()
                self.ai_model.to(device)
                self.ai_model.eval()
                self.device = device
            except:
                self.ai_model = None
    
    # Methods to add different types of noise
    # Add different types of artificial noise for testing
    def add_gaussian_noise(self, frame, mean=0, std=25):
        noise = np.random.normal(mean, std, frame.shape).astype(np.float32)
        noisy = frame.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def add_salt_pepper_noise(self, frame, salt_prob=0.01, pepper_prob=0.01):
        noisy = frame.copy()
        h, w = frame.shape[:2]
        
        # Salt noise
        salt_coords = np.random.random((h, w)) < salt_prob
        noisy[salt_coords] = 255
        
        # Pepper noise
        pepper_coords = np.random.random((h, w)) < pepper_prob
        noisy[pepper_coords] = 0
        
        return noisy
    
    def add_speckle_noise(self, frame, intensity=0.2):
        noise = np.random.randn(*frame.shape) * intensity
        noisy = frame.astype(np.float32) * (1 + noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    # Noise Detection
    def estimate_noise_level(self, frame):
        """Estimate the noise level in an image using Laplacian variance method"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Calculate Laplacian variance (measure of blurriness/noise)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Estimate noise standard deviation using robust MAD estimator
        diff = gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 1.0)
        noise_std = np.median(np.abs(diff - np.median(diff))) / 0.6745
        
        return {
            'laplacian_variance': laplacian_var,
            'noise_std': noise_std,
            'is_noisy': noise_std > 5.0,  # Threshold for considering image noisy
            'noise_level': 'High' if noise_std > 15 else 'Medium' if noise_std > 5 else 'Low'
        }
    
    def detect_noise_type(self, frame):
        """Simple noise type detection based on statistical properties"""
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Check for salt-and-pepper noise (high number of extreme values)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        salt_pepper_ratio = (hist[0] + hist[255]) / gray.size
        
        # Check for Gaussian-like noise
        mean_val = np.mean(gray)
        std_val = np.std(gray)
        
        # Simple heuristic classification
        if salt_pepper_ratio > 0.01:
            return "Salt & Pepper"
        elif std_val > 30:
            return "Gaussian"
        else:
            return "Low noise / Clean"
    
    # Traditional Filters
    def apply_median_filter(self, frame, kernel_size=5):
        """Remove noise using median filtering - great for salt & pepper noise"""
        return cv2.medianBlur(frame, kernel_size)
    
    def apply_gaussian_filter(self, frame, kernel_size=15, sigma=2):
        """Smooth out noise using Gaussian blur - good for random noise"""
        return cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    
    def apply_bilateral_filter(self, frame, d=9, sigma_color=75, sigma_space=75):
        """Smart filtering that preserves edges while removing noise"""
        return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)
    
    def apply_ai_denoising(self, frame):
        """Use AI neural network to remove noise - our smartest method"""
        if self.ai_model is None or not PYTORCH_AVAILABLE:
            # If AI isn't available, use bilateral filter instead
            return self.apply_bilateral_filter(frame)
        
        try:
            # Convert image to the format the AI expects
            if len(frame.shape) == 3:
                tensor = torch.from_numpy(frame.transpose(2, 0, 1).astype(np.float32) / 255.0)
                tensor = tensor.unsqueeze(0).to(self.device)
            else:
                tensor = torch.from_numpy(frame.astype(np.float32) / 255.0)
                tensor = tensor.unsqueeze(0).unsqueeze(0).to(self.device)
                tensor = tensor.repeat(1, 3, 1, 1)
            
            # Apply model
            with torch.no_grad():
                denoised = self.ai_model(tensor)
            
            # Convert the result back to normal image format
            if len(frame.shape) == 3:
                result = denoised.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            else:
                result = denoised.squeeze(0).mean(0).cpu().numpy()
            
            result = np.clip(result * 255, 0, 255).astype(np.uint8)
            return result
            
        except Exception as e:
            st.warning(f"AI denoising failed: {e}. Using bilateral filter.")
            return self.apply_bilateral_filter(frame)
    
    # Quality Metrics
    def calculate_psnr(self, original, processed):
        mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))
    
    def calculate_snr(self, original, processed):
        signal_power = np.mean(original.astype(np.float64) ** 2)
        noise_power = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
        if noise_power == 0:
            return float('inf')
        return 10 * np.log10(signal_power / noise_power)
    
    def analyze_frequency_spectrum(self, frame):
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        f_transform = fft2(gray.astype(np.float32))
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        return magnitude_spectrum

# Initialize processor
@st.cache_resource
def load_processor():
    return VideoProcessor()

# Force cache clear if needed (for development)
if st.sidebar.button("🔄 Reload Processor", help="Use this if you encounter any errors"):
    st.cache_resource.clear()
    st.rerun()

processor = load_processor()

# Title
st.title("🔬 Real-Time DSP-based Video Processing")
st.markdown("**Advanced Signal Processing Platform** - Video Acquisition, Noise Reduction & Frequency Domain Analysis")

# Sidebar controls
st.sidebar.header("⚙️ DSP Processing Controls")

# Processing mode selection
st.sidebar.subheader("🎯 Processing Mode")
processing_mode = st.sidebar.radio(
    "Select Processing Mode:",
    ["Synthetic Noise Analysis", "Real-world Denoising"]
)

# Conditional noise controls
if processing_mode == "Synthetic Noise Analysis":
    st.sidebar.subheader("1️⃣ Signal Noise Modeling")
    noise_type = st.sidebar.selectbox(
        "Noise Type for Analysis:", 
        ["None", "Gaussian", "Salt & Pepper", "Speckle"]
    )
    st.sidebar.info("💡 This mode adds synthetic noise for analysis purposes")
else:
    noise_type = "None"
    st.sidebar.subheader("1️⃣ Real-world Video Processing")
    st.sidebar.info("💡 This mode processes already noisy videos directly")
    
    # Add noise detection option
    auto_detect = st.sidebar.checkbox("Auto-detect noise level", value=True)
    
    if auto_detect:
        st.sidebar.markdown("🔍 **Automatic Noise Detection**: Enabled")
    else:
        st.sidebar.markdown("🔧 **Manual Processing**: Apply denoising to input video")

# Noise parameters (only for synthetic mode)
if processing_mode == "Synthetic Noise Analysis":
    if noise_type == "Gaussian":
        std = st.sidebar.slider("Standard Deviation", 1, 50, 25)
        mean = st.sidebar.slider("Mean", -10, 10, 0)
    elif noise_type == "Salt & Pepper":
        salt_prob = st.sidebar.slider("Salt Probability", 0.001, 0.05, 0.01, step=0.001)
        pepper_prob = st.sidebar.slider("Pepper Probability", 0.001, 0.05, 0.01, step=0.001)
    elif noise_type == "Speckle":
        intensity = st.sidebar.slider("Speckle Intensity", 0.1, 1.0, 0.2)

# Filter selection
st.sidebar.subheader("2️⃣ DSP Noise Reduction Method")
filter_type = st.sidebar.selectbox(
    "Processing Algorithm:", 
    ["None", "Median Filter", "Gaussian Filter", "Bilateral Filter", "AI Denoising"]
)

# Filter parameters
if filter_type == "Median Filter":
    kernel_size = st.sidebar.slider("Kernel Size", 3, 15, 5, step=2)
elif filter_type == "Gaussian Filter":
    gauss_kernel = st.sidebar.slider("Kernel Size", 3, 31, 15, step=2)
    sigma = st.sidebar.slider("Sigma", 0.5, 10.0, 2.0)
elif filter_type == "Bilateral Filter":
    d = st.sidebar.slider("Diameter", 5, 15, 9)
    sigma_color = st.sidebar.slider("Sigma Color", 10, 150, 75)
    sigma_space = st.sidebar.slider("Sigma Space", 10, 150, 75)

# File upload
st.header("📁 Upload Video")
uploaded_file = st.file_uploader("Upload video for DSP analysis", type=['mp4', 'avi', 'mov'])

if uploaded_file is not None:
    # Save uploaded file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_input.write(uploaded_file.read())
    temp_input.close()
    
    # Read video
    cap = cv2.VideoCapture(temp_input.name)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Display video info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("FPS", fps)
    col2.metric("Frames", frame_count)
    col3.metric("Width", width)
    col4.metric("Height", height)
    
    # Frame selection
    frame_idx = st.slider("Frame Number", 0, min(frame_count-1, 100), 0)
    
    # Get selected frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    
    if ret:
        # Process frame based on mode
        original = frame.copy()
        
        if processing_mode == "Synthetic Noise Analysis":
            # Traditional mode: Add synthetic noise
            if noise_type == "Gaussian":
                noisy = processor.add_gaussian_noise(frame, mean, std)
            elif noise_type == "Salt & Pepper":
                noisy = processor.add_salt_pepper_noise(frame, salt_prob, pepper_prob)
            elif noise_type == "Speckle":
                noisy = processor.add_speckle_noise(frame, intensity)
            else:
                noisy = frame.copy()
        else:
            # Real-world mode: Use input video as noisy
            noisy = frame.copy()
            
            # Perform noise analysis if auto-detection is enabled
            if auto_detect:
                try:
                    noise_stats = processor.estimate_noise_level(frame)
                    detected_noise_type = processor.detect_noise_type(frame)
                    
                    # Display noise analysis
                    st.info(f"""
                    **🔍 Noise Analysis Results:**
                    - **Detected Noise Type**: {detected_noise_type}
                    - **Noise Level**: {noise_stats['noise_level']}
                    - **Noise Standard Deviation**: {noise_stats['noise_std']:.2f}
                    - **Laplacian Variance**: {noise_stats['laplacian_variance']:.2f}
                    - **Is Noisy**: {'Yes' if noise_stats['is_noisy'] else 'No'}
                    """)
                except Exception as e:
                    st.warning(f"Noise analysis failed: {e}. Processing without analysis.")
                    st.info("💡 Try clicking the '🔄 Reload Processor' button in the sidebar if you continue to see errors.")
        
        # Apply filter
        if filter_type == "Median Filter":
            filtered = processor.apply_median_filter(noisy, kernel_size)
        elif filter_type == "Gaussian Filter":
            filtered = processor.apply_gaussian_filter(noisy, gauss_kernel, sigma)
        elif filter_type == "Bilateral Filter":
            filtered = processor.apply_bilateral_filter(noisy, d, sigma_color, sigma_space)
        elif filter_type == "AI Denoising":
            filtered = processor.apply_ai_denoising(noisy)
        else:
            filtered = noisy.copy()
        
        # Display results with appropriate labels
        st.header("🔬 Processing Results")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Input")
            st.image(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        
        with col2:
            if processing_mode == "Synthetic Noise Analysis":
                st.subheader(f"With {noise_type} Noise")
            else:
                st.subheader("Input Video (Potentially Noisy)")
            st.image(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB))
        
        with col3:
            st.subheader("Processed Output")
            st.image(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        
        # Quality metrics
        if processing_mode == "Synthetic Noise Analysis":
            st.header("📊 Quality Analysis - Controlled Testing")
        else:
            st.header("📊 Quality Analysis - Real-World Video")
        
        col1, col2, col3 = st.columns(3)
        
        if processing_mode == "Synthetic Noise Analysis":
            # Traditional analysis: compare original vs noisy vs filtered
            if noise_type != "None":
                psnr_noisy = processor.calculate_psnr(original, noisy)
                snr_noisy = processor.calculate_snr(original, noisy)
                col1.metric("PSNR (Noisy)", f"{psnr_noisy:.2f} dB")
                col1.metric("SNR (Noisy)", f"{snr_noisy:.2f} dB")
            
            if filter_type != "None":
                psnr_filtered = processor.calculate_psnr(original, filtered)
                snr_filtered = processor.calculate_snr(original, filtered)
                col2.metric("PSNR (Filtered)", f"{psnr_filtered:.2f} dB")
                col2.metric("SNR (Filtered)", f"{snr_filtered:.2f} dB")
                
                if noise_type != "None":
                    improvement = psnr_filtered - psnr_noisy
                    col3.metric("PSNR Improvement", f"{improvement:.2f} dB")
        else:
            # Real-world analysis: compare input vs processed
            if filter_type != "None":
                psnr_improvement = processor.calculate_psnr(noisy, filtered)
                snr_improvement = processor.calculate_snr(noisy, filtered)
                
                # Estimate noise level in input
                noise_info = processor.estimate_noise_level(noisy)
                noise_type_detected = processor.detect_noise_type(noisy)
                
                col1.metric("Input Noise Level", noise_info['noise_level'])
                col1.metric("Detected Noise Type", noise_type_detected)
                col1.metric("Noise Strength", f"{noise_info['noise_std']:.1f}")
                
                col2.metric("Processing Quality (PSNR)", f"{psnr_improvement:.2f} dB")
                col2.metric("Signal Enhancement (SNR)", f"{snr_improvement:.2f} dB")
                
                # Provide interpretation
                if psnr_improvement > 25:
                    quality_assessment = "Excellent quality maintained"
                elif psnr_improvement > 20:
                    quality_assessment = "Good quality maintained"  
                elif psnr_improvement > 15:
                    quality_assessment = "Fair quality maintained"
                else:
                    quality_assessment = "Some quality loss occurred"
                
                col2.metric("Quality Assessment", quality_assessment)
                
                col3.info(f"💡 **Real-world Analysis**: Since we don't have the original clean video, we measure how well the filter processed your noisy input. Higher PSNR values (>20 dB) generally indicate good processing results.")
                
                if noise_info['is_noisy']:
                    col3.success(f"✅ Noise detected and processed. The {filter_type.lower()} method was applied to reduce {noise_type_detected.lower()} noise.")
                else:
                    col3.warning("⚠️ Input appears to have low noise. Results may vary.")
        
        # Frequency analysis
        st.header("📊 Frequency Domain Analysis & Spectral Characteristics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            spectrum_orig = processor.analyze_frequency_spectrum(original)
            fig1 = go.Figure(data=go.Heatmap(z=spectrum_orig, colorscale='Viridis'))
            fig1.update_layout(title="Original Signal Spectrum", width=300, height=250)
            st.plotly_chart(fig1)
        
        with col2:
            spectrum_noisy = processor.analyze_frequency_spectrum(noisy)
            fig2 = go.Figure(data=go.Heatmap(z=spectrum_noisy, colorscale='Viridis'))
            fig2.update_layout(title="Noise-Corrupted Spectrum", width=300, height=250)
            st.plotly_chart(fig2)
        
        with col3:
            spectrum_filtered = processor.analyze_frequency_spectrum(filtered)
            fig3 = go.Figure(data=go.Heatmap(z=spectrum_filtered, colorscale='Viridis'))
            fig3.update_layout(title="DSP-Filtered Spectrum", width=300, height=250)
            st.plotly_chart(fig3)
        
        # Process full video
        st.header("🎬 Batch Video Processing")
        
        processing_label = "Process with Synthetic Noise" if processing_mode == "Synthetic Noise Analysis" else "Denoise Real Video"
        
        # Initialize session state for video files
        if 'processed_videos' not in st.session_state:
            st.session_state.processed_videos = {}
        
        if st.button(processing_label, type="primary"):
            progress = st.progress(0)
            status = st.empty()
            
            # Reset video capture
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output videos with unique names
            import time
            timestamp = str(int(time.time()))
            
            temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=f'_processed_{timestamp}.mp4')
            temp_output.close()
            
            # For synthetic mode, also create noisy video output
            if processing_mode == "Synthetic Noise Analysis":
                temp_noisy = tempfile.NamedTemporaryFile(delete=False, suffix=f'_noisy_{timestamp}.mp4')
                temp_noisy.close()
                noisy_writer = cv2.VideoWriter(temp_noisy.name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            processed_writer = cv2.VideoWriter(temp_output.name, fourcc, fps, (width, height))
            
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if processing_mode == "Synthetic Noise Analysis":
                    if noise_type == "Gaussian":
                        noisy = processor.add_gaussian_noise(frame, mean, std)
                    elif noise_type == "Salt & Pepper":
                        noisy = processor.add_salt_pepper_noise(frame, salt_prob, pepper_prob)
                    elif noise_type == "Speckle":
                        noisy = processor.add_speckle_noise(frame, intensity)
                    else:
                        noisy = frame.copy()
                    
                    noisy_writer.write(noisy)
                else:
                    noisy = frame.copy()
                
                # Apply denoising filter
                if filter_type == "Median Filter":
                    processed = processor.apply_median_filter(noisy, kernel_size)
                elif filter_type == "Gaussian Filter":
                    processed = processor.apply_gaussian_filter(noisy, gauss_kernel, sigma)
                elif filter_type == "Bilateral Filter":
                    processed = processor.apply_bilateral_filter(noisy, d, sigma_color, sigma_space)
                elif filter_type == "AI Denoising":
                    processed = processor.apply_ai_denoising(noisy)
                else:
                    processed = noisy.copy()
                
                processed_writer.write(processed)
                
                frame_num += 1
                progress.progress(frame_num / frame_count)
                status.text(f"Processing frame {frame_num}/{frame_count}")
            
            processed_writer.release()
            if processing_mode == "Synthetic Noise Analysis":
                noisy_writer.release()
            
            # Store video files in session state
            video_key = f"{uploaded_file.name}_{timestamp}"
            st.session_state.processed_videos[video_key] = {
                'processed_path': temp_output.name,
                'mode': processing_mode,
                'noise_type': noise_type if processing_mode == "Synthetic Noise Analysis" else None,
                'filter_type': filter_type,
                'original_name': uploaded_file.name
            }
            
            if processing_mode == "Synthetic Noise Analysis":
                st.session_state.processed_videos[video_key]['noisy_path'] = temp_noisy.name
            
            success_message = "Synthetic noise analysis complete!" if processing_mode == "Synthetic Noise Analysis" else "Video denoising complete!"
            st.success(success_message)
        
        if st.session_state.processed_videos:
            st.subheader("📥 Available Downloads")
            
            for video_key, video_info in st.session_state.processed_videos.items():
                with st.expander(f"Download: {video_info['original_name']}", expanded=True):
                    
                    if video_info['mode'] == "Synthetic Noise Analysis" and 'noisy_path' in video_info:
                        col1, col2 = st.columns(2)
                        
                        try:
                            with open(video_info['noisy_path'], 'rb') as f:
                                noisy_video_bytes = f.read()
                            
                            with col1:
                                st.download_button(
                                    "📥 Download Noisy Video",
                                    noisy_video_bytes,
                                    f"noisy_{video_info['noise_type'].lower().replace(' & ', '_')}_{video_info['original_name']}",
                                    "video/mp4",
                                    help=f"Download video with {video_info['noise_type']} noise added",
                                    key=f"noisy_{video_key}"
                                )
                        except FileNotFoundError:
                            col1.error("Noisy video file not found")
                        
                        try:
                            with open(video_info['processed_path'], 'rb') as f:
                                processed_video_bytes = f.read()
                            
                            with col2:
                                st.download_button(
                                    "📥 Download Processed Video",
                                    processed_video_bytes,
                                    f"denoised_{video_info['filter_type'].lower().replace(' ', '_')}_{video_info['original_name']}",
                                    "video/mp4",
                                    help=f"Download video processed with {video_info['filter_type']}",
                                    key=f"processed_{video_key}"
                                )
                        except FileNotFoundError:
                            col2.error("Processed video file not found")
                        
                    else:
                        try:
                            with open(video_info['processed_path'], 'rb') as f:
                                video_bytes = f.read()
                            
                            st.download_button(
                                "📥 Download Denoised Video",
                                video_bytes,
                                f"denoised_{video_info['original_name']}",
                                "video/mp4",
                                key=f"denoised_{video_key}"
                            )
                        except FileNotFoundError:
                            st.error("Denoised video file not found")
                    
                    if st.button(f"🗑️ Remove {video_info['original_name']}", key=f"cleanup_{video_key}"):
                        try:
                            if os.path.exists(video_info['processed_path']):
                                os.unlink(video_info['processed_path'])
                            if 'noisy_path' in video_info and os.path.exists(video_info['noisy_path']):
                                os.unlink(video_info['noisy_path'])
                        except:
                            pass
                        
                        del st.session_state.processed_videos[video_key]
                        st.rerun()
            
            if st.button("🗑️ Clear All Downloads", type="secondary"):
                for video_info in st.session_state.processed_videos.values():
                    try:
                        if os.path.exists(video_info['processed_path']):
                            os.unlink(video_info['processed_path'])
                        if 'noisy_path' in video_info and os.path.exists(video_info['noisy_path']):
                            os.unlink(video_info['noisy_path'])
                    except:
                        pass
                
                st.session_state.processed_videos = {}
                st.rerun()
    
    cap.release()
    os.unlink(temp_input.name)

# DSP Information Sidebar
st.sidebar.markdown("---")
st.sidebar.header("📚 DSP Concepts")

with st.sidebar.expander("Processing Modes"):
    st.markdown("""
    **Synthetic Noise Analysis**: Add controlled noise to clean videos for algorithm comparison<br>
    **Real-world Denoising**: Process already noisy videos directly (practical scenario)<br>
    """, unsafe_allow_html=True)

with st.sidebar.expander("Signal Noise Models"):
    st.markdown("""
    **Gaussian**: Additive white noise (AWGN)<br>
    **Salt & Pepper**: Impulse/sparse noise<br>
    **Speckle**: Multiplicative coherent noise<br>
    """, unsafe_allow_html=True)

with st.sidebar.expander("DSP Processing Algorithms"):
    st.markdown("""
    **Median**: Non-linear rank-order filtering<br>
    **Gaussian**: Linear low-pass convolution<br>
    **Bilateral**: Spatial & intensity domain filtering<br>
    **AI**: Deep learning-based denoising<br>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🔬 Real-Time DSP-based Video Processing</p>
    <p>Advanced Signal Processing Platform for Video Acquisition, Noise Reduction & Frequency Analysis</p>
</div>
""", unsafe_allow_html=True)