import os
import json
import glob
import shutil
from pathlib import Path

def process_json_files(input_dir, output_dir, batch_size=1000):
    """
    Process multiple JSON files and organize them into batches.
    
    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save processed files
        batch_size: Number of files per batch
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(input_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    # Process files in batches
    for i, file_path in enumerate(json_files):
        # Determine batch directory
        batch_num = i // batch_size
        batch_dir = os.path.join(output_dir, f"batch_{batch_num}")
        os.makedirs(batch_dir, exist_ok=True)
        
        # Process file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Standardize annotations
            if "annotations" in data:
                standardized_annotations = []
                for annotation in data["annotations"]:
                    if len(annotation) >= 3:
                        start, end, label = annotation[0], annotation[1], annotation[2]
                        
                        # Extract entity type from label
                        if ":" in label:
                            entity_type = label.split(":")[0].strip()
                        else:
                            entity_type = label
                        
                        standardized_annotations.append([start, end, entity_type])
                
                data["annotations"] = standardized_annotations
            
            # Save processed file
            output_file = os.path.join(batch_dir, os.path.basename(file_path))
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} files")
        
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    print(f"Processed {len(json_files)} files into {(len(json_files) // batch_size) + 1} batches")

def merge_batches(batch_dir, output_file):
    """
    Merge processed batches into a single training file.
    
    Args:
        batch_dir: Directory containing batch directories
        output_file: Path to save the merged file
    """
    all_data = []
    
    # Get all batch directories
    batch_dirs = [d for d in os.listdir(batch_dir) if os.path.isdir(os.path.join(batch_dir, d)) and d.startswith("batch_")]
    
    for batch in batch_dirs:
        batch_path = os.path.join(batch_dir, batch)
        json_files = glob.glob(os.path.join(batch_path, "*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_data.append(data)
            except Exception as e:
                print(f"Error reading {file_path}: {str(e)}")
    
    # Save merged data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=2)
    
    print(f"Merged {len(all_data)} examples into {output_file}")

def main():
    # Parameters
    raw_data_dir = "data/raw"
    processed_data_dir = "data/processed"
    merged_file = "data/all_training_data.json"
    
    # Process JSON files
    process_json_files(raw_data_dir, processed_data_dir)
    
    # Merge batches
    merge_batches(processed_data_dir, merged_file)

if __name__ == "__main__":
    main()