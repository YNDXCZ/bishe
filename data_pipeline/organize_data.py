import os
import shutil
import keyword

def organize_dataset(source_dir, target_dir="data/raw"):
    """
    Scans source_dir for images. 
    If path contains 'good', 'upright' -> copy to target/good
    If path contains 'bad', 'hunch', 'slouch' -> copy to target/bad
    """
    print(f"Scanning {source_dir}...")
    
    good_keywords = ['good', 'upright', 'straight', 'correct']
    bad_keywords = ['bad', 'slouch', 'hunch', 'lean', 'incorrect']
    
    counts = {'good': 0, 'bad': 0}
    
    os.makedirs(os.path.join(target_dir, 'good'), exist_ok=True)
    os.makedirs(os.path.join(target_dir, 'bad'), exist_ok=True)
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            full_path = os.path.join(root, file)
            lower_path = full_path.lower()
            
            # Determine category
            category = None
            if any(k in lower_path for k in good_keywords):
                category = 'good'
            elif any(k in lower_path for k in bad_keywords):
                category = 'bad'
            
            if category:
                # Copy file
                dest = os.path.join(target_dir, category, f"imported_{counts[category]}_{file}")
                shutil.copy2(full_path, dest)
                counts[category] += 1
                
    print(f"Organization Complete.")
    print(f"Imported Good: {counts['good']}")
    print(f"Imported Bad: {counts['bad']}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        source = sys.argv[1]
    else:
        source = input("Enter path to downloaded dataset folder: ")
    
    if os.path.exists(source):
        organize_dataset(source)
    else:
        print("Path does not exist.")
