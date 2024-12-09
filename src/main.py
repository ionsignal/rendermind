import sys
import traceback
import argparse
import torch
from .rendermind.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="Video upscaling using restoration models")
    parser.add_argument("-i", "--input", 
        type=str, 
        required=True,
        help="Input video path")
    parser.add_argument("-o", "--output", 
        type=str, 
        required=True,
        help="Output video path")
    args = parser.parse_args()

    # Add exit code tracking
    exit_code = 0

    try:
        # instantiate processor
        processor = VideoProcessor(
            devices="cuda:0,cuda:1,cuda:2,cuda:3",
        )
        # process video
        processor.process_video(
            input_path=args.input,
            output_path=args.output
        )

    except Exception as e:
        print(f"Processing failed with error: {str(e)}")
        print("\nFull stack trace:")
        traceback.print_exc()
        exit_code = 1
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        sys.exit(exit_code)

if __name__ == "__main__":
    main()