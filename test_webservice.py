import requests

# Base URL of the API
BASE_URL = 'http://127.0.0.1:5000'

def get_speakers():
    """Get list of available speakers"""
    response = requests.get(f'{BASE_URL}/speakers')
    return response.json()

def generate_speech(content, prompt, speaker):
    """Generate speech from text"""
    data = {
        'content': content,
        'prompt': prompt,
        'speaker': speaker
    }
    response = requests.post(
        f'{BASE_URL}/generate',
        json=data,
        stream=True
    )
    
    if response.status_code == 200:
        # Save the audio file
        with open('outputs/generated_speech.wav', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    else:
        print(f"Error: {response.status_code}")
        print(response.json())
        return False

def main():
    # Get available speakers
    speakers = get_speakers()
    print("Available speakers:", speakers)
    
    # Test speech generation
    content = "Hello, this is a test."
    prompt = "happy"
    speaker = speakers[0]
    
    success = generate_speech(content, prompt, speaker)
    if success:
        print("Speech generated successfully! Check 'generated_speech.wav'")

if __name__ == "__main__":
    main()