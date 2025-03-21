import os

def test_output_generated():
    # Run the main script
    import main
    main.main()
    
    # Check if output file exists
    assert os.path.exists('output_visual_effect.jpg'), "Output file was not generated"
    
    # Check if file size is reasonable (not empty)
    assert os.path.getsize('output_visual_effect.jpg') > 1000, "Output file seems to be empty or corrupted"

if __name__ == '__main__':
    test_output_generated()