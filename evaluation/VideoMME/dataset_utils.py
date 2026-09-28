import os
import torch
import string
import pandas as pd
from typing import Dict, Any, List
from datasets import load_dataset

def load_videomme_dataset(data_dir, duration='short'):
    """
    Load the VideoMME dataset.
    
    Args:
        data_dir: Directory containing VideoMME data
        duration: Video duration type ('short', 'medium', or 'long')
    
    Returns:
        List of data samples
    """
    print(f"Loading VideoMME dataset with duration={duration}")
    
    total_data = []
    for item in load_dataset(data_dir)["test"]:
        if item['duration'] == duration:
            total_data.append(item)
    
    print(f"✓ Loaded {len(total_data)} samples with duration={duration}")
    return total_data

def extract_video_frames_with_timestamps(video_path, fps=2, min_frames=4, max_frames=512):
    """
    Extract frames from video and return their timestamps.
    
    Args:
        video_path: Path to video file
        fps: Frames per second to extract
        min_frames: Minimum number of frames
        max_frames: Maximum number of frames
    
    Returns:
        Tuple of (frame_indices, frame_timestamps)
    """
    from decord import VideoReader
    
    video_reader = VideoReader(video_path, num_threads=1)
    video_len = len(video_reader)
    duration = video_len / video_reader.get_avg_fps()
    
    # Calculate number of frames to extract
    nframes = round(duration) * fps
    nframes = min(max(nframes, min_frames), max_frames, video_len // 2 * 2)
    
    # Extract frame indices
    indices = torch.linspace(0, video_len - 1, nframes).round().long().clamp(0, video_len - 1).tolist()
    
    # Get frame timestamps
    frame_timestamps = video_reader.get_frame_timestamp(indices)[:, 0].tolist()
    
    return indices, frame_timestamps

def load_subtitles(subtitle_path, frame_timestamps):
    """
    Load subtitles and match them to video frames.
    
    Args:
        subtitle_path: Path to .srt subtitle file
        frame_timestamps: List of frame timestamps in seconds
    
    Returns:
        String of matched subtitles
    """
    import pysubs2
    
    if not os.path.exists(subtitle_path):
        return ""
    
    subs = pysubs2.load(subtitle_path, encoding='utf-8')
    subtitles = []
    
    for sub in subs:
        for frame_timestamp in frame_timestamps:
            if sub.start / 1000 <= frame_timestamp and sub.end / 1000 > frame_timestamp:
                sub_text = sub.text.replace('\\N', ' ')
                if sub_text.strip():
                    subtitles.append(sub_text)
                    break
    
    return ' '.join(subtitles)

def build_videomme_prompt(data, data_dir, use_subtitle=False, fps=2, 
                          min_frames=4, max_frames=512, 
                          min_pixels=128*28*28, max_pixels=512*28*28, 
                          total_pixels=24576*28*28, sys_prompt=None):
    """
    Build VideoMME prompt (consistent with original implementation).
    
    Args:
        data: Single data sample
        data_dir: VideoMME data directory
        use_subtitle: Whether to include subtitles
        fps: Frames per second
        min_frames: Minimum frames
        max_frames: Maximum frames
        min_pixels: Minimum pixels per frame
        max_pixels: Maximum pixels per frame
        total_pixels: Total pixels across all frames
        sys_prompt: Optional system prompt
    
    Returns:
        Tuple of (messages, annotation)
    """
    video_id = data['videoID']
    duration = data['duration']
    domain = data['domain']
    sub_category = data["sub_category"]
    question = data['question']
    choices = data['options']
    answer = data['answer']
    question_id = data['question_id']
    
    video_path = os.path.join(data_dir, 'videos', f'{video_id}.mp4')
    subtitle_path = os.path.join(data_dir, 'subtitle', f'{video_id}.srt')
    
    # Build choices text
    choice_txt = '\n'.join(choices)
    
    # Build prompt
    prompt = ''
    if use_subtitle and os.path.exists(subtitle_path):
        # Extract frame timestamps
        _, frame_timestamps = extract_video_frames_with_timestamps(
            video_path, fps=fps, min_frames=min_frames, max_frames=max_frames
        )
        
        # Load and match subtitles
        subtitles = load_subtitles(subtitle_path, frame_timestamps)
        
        if subtitles:
            prompt = "This video's subtitles are listed below:\n"
            prompt += subtitles + '\n'
    
    prompt += 'Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.'
    prompt += f"\nQuestion: {question}\n{choice_txt}\nThe best answer is:"
    
    # Build video content
    video_content = {
        "video": video_path,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
        "min_frames": min_frames,
        "max_frames": max_frames,
        "total_pixels": total_pixels,
        "fps": fps
    }
    
    contents = [
        video_content,
        {
            "text": prompt
        }
    ]
    
    # Build messages
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    
    messages.append({
        "role": "user",
        "content": contents
    })
    
    # Build annotation
    assert answer in ['A', 'B', 'C', 'D', 'E']
    answer_id = ord(answer) - 65
    
    annotation = {
        "question": question,
        "choices": {
            string.ascii_uppercase[i]: choice.split(".", 1)[1].strip() 
            for i, choice in enumerate(choices)
        },
        "answer": answer,
        "answer_id": answer_id,
        "video_path": video_path,
        "domain": domain,
        "sub_category": sub_category,
        "question_id": question_id
    }
    
    return messages, annotation

