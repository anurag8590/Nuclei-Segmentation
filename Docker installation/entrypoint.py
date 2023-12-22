import sys
from inference import inference

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python entrypoint.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    inference(input_path, output_path)