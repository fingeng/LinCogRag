"""
Download biomedical-ner-all model from Hugging Face
"""
from huggingface_hub import snapshot_download
import os

def download_model():
    """Download the biomedical NER model"""
    model_name = "d4data/biomedical-ner-all"
    local_dir = "models/biomedical-ner-all"
    
    print("="*70)
    print("Downloading Biomedical NER Model")
    print("="*70)
    print(f"\nModel: {model_name}")
    print(f"Target directory: {local_dir}")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        print("\n📥 Downloading... (this may take a few minutes)")
        
        snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Don't use symlinks
            resume_download=True,  # Resume if interrupted
        )
        
        print("\n✅ Download complete!")
        print(f"\nModel saved to: {os.path.abspath(local_dir)}")
        
        # List downloaded files
        print("\n📂 Downloaded files:")
        for file in os.listdir(local_dir):
            file_path = os.path.join(local_dir, file)
            size = os.path.getsize(file_path) / (1024*1024)  # MB
            print(f"   • {file:40s} ({size:.2f} MB)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False

if __name__ == '__main__':
    success = download_model()
    
    if success:
        print("\n✅ Model ready to use!")
        print("\nNext step: Test the model")
        print("   python test_enhanced_ner_standalone.py")
    else:
        print("\n❌ Download failed!")
