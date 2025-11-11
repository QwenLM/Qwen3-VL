import json
import os
from tqdm import tqdm
from datasets import load_dataset

def validate_data(json_file_path, media_folder_path):
    """
    Main function to validate JSON data by checking:
    1. Media file existence (supports both image and video fields)
    2. Media token consistency in conversations
    Saves valid and problematic data to separate files
    """
    # Validate input file format
    if not json_file_path.endswith((".json", ".jsonl")):
        print("Invalid file format. Please provide a .json or .jsonl file.")
        return
    
    # Prepare output file paths
    base_path = os.path.splitext(json_file_path)[0]
    valid_file_path = f"{base_path}_valid.json"
    problem_file_path = f"{base_path}_problems.json"
    
    # Load the dataset
    try:
        data = load_dataset("json", data_files=json_file_path)["train"]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    valid_data = []
    problem_data = []
    stats = {
        'total_entries': 0,
        'valid_entries': 0,
        'missing_media': 0,
        'token_mismatches': 0,
        'gpt_media_tokens': 0,
        'missing_files': [],
        'media_types': {
            'image': 0,
            'video': 0,
            'mixed': 0
        }
    }
    
    print(f"Processing {len(data)} entries...")
    
    for item in tqdm(data):
        stats['total_entries'] += 1
        problems = []
        
        # Check media file existence (handle both singular and plural fields)
        media_info = {
            'image': item.get("image", item.get("images", [])),
            'video': item.get("video", item.get("videos", []))
        }
        
        # Convert all media fields to lists
        for media_type in media_info:
            if isinstance(media_info[media_type], str):
                media_info[media_type] = [media_info[media_type]]
            elif not isinstance(media_info[media_type], list):
                media_info[media_type] = []
        
        # Count media types for stats
        media_counts = {k: len(v) for k, v in media_info.items()}
        active_media = [k for k, v in media_counts.items() if v > 0]
        
        if len(active_media) > 1:
            stats['media_types']['mixed'] += 1
        elif len(active_media) == 1:
            stats['media_types'][active_media[0]] += 1
        
        # Check all media files exist
        missing_files = []
        for media_type, files in media_info.items():
            for media_file in files:
                media_path = os.path.join(media_folder_path, media_file)
                if not os.path.exists(media_path):
                    missing_files.append(media_path)
        
        if missing_files:
            stats['missing_media'] += 1
            stats['missing_files'].extend(missing_files)
            problems.append({
                'type': 'missing_files',
                'files': missing_files,
                'message': f"Missing media files: {missing_files}"
            })
        
        # Check media token consistency
        conversations = item.get("conversations", [])
        expected_counts = {
            'image': media_counts['image'],
            'video': media_counts['video']
        }
        
        actual_counts = {
            'image': 0,
            'video': 0
        }
        gpt_has_media_token = False
        
        for conv in conversations:
            if conv.get("from") == "human":
                actual_counts['image'] += conv.get("value", "").count("<image>")
                actual_counts['video'] += conv.get("value", "").count("<video>")
            elif conv.get("from") == "gpt":
                if "<image>" in conv.get("value", "") or "<video>" in conv.get("value", ""):
                    gpt_has_media_token = True
        
        # Check token counts match media counts
        for media_type in ['image', 'video']:
            if actual_counts[media_type] != expected_counts[media_type]:
                stats['token_mismatches'] += 1
                problems.append({
                    'type': 'token_mismatch',
                    'media_type': media_type,
                    'expected': expected_counts[media_type],
                    'actual': actual_counts[media_type],
                    'message': f"Expected {expected_counts[media_type]} <{media_type}> tokens, found {actual_counts[media_type]}"
                })
                break  # Count each entry only once for mismatches
        
        if gpt_has_media_token:
            stats['gpt_media_tokens'] += 1
            problems.append({
                'type': 'gpt_media_token',
                'message': "GPT response contains media token (<image> or <video>)"
            })
        
        # Categorize the item
        if not problems:
            stats['valid_entries'] += 1
            valid_data.append(item)
        else:
            problem_item = item.copy()
            problem_item['validation_problems'] = problems
            problem_data.append(problem_item)
    
    # Save results
    with open(valid_file_path, 'w') as f:
        json.dump(valid_data, f, indent=2)
    
    with open(problem_file_path, 'w') as f:
        json.dump(problem_data, f, indent=2)
    
    # Print summary
    print("\nValidation Summary:")
    print(f"Total entries processed: {stats['total_entries']}")
    print(f"Valid entries: {stats['valid_entries']} ({stats['valid_entries']/stats['total_entries']:.1%})")
    print(f"Media type distribution:")
    print(f"  - Image only: {stats['media_types']['image']}")
    print(f"  - Video only: {stats['media_types']['video']}")
    print(f"  - Mixed media: {stats['media_types']['mixed']}")
    print(f"Entries with missing media: {stats['missing_media']}")
    print(f"Entries with token mismatches: {stats['token_mismatches']}")
    print(f"Entries with GPT media tokens: {stats['gpt_media_tokens']}")
    
    if stats['missing_files']:
        print("\nSample missing files (max 5):")
        for f in stats['missing_files'][:5]:
            print(f"  - {f}")

# Example usage
if __name__ == "__main__":
    json_file_path = "example.json"  # Replace with your JSON file path
    media_folder_path = "media"      # Replace with your media folder path
    validate_data(json_file_path, media_folder_path)