import sounddevice as sd
import soundfile as sf
import requests
import pyttsx3
import streamlit as st
import numpy as np

# API setup
API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v2"
headers = {"Authorization": "Bearer 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"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def record_audio():
    sd.default.device = None 
    device_info = sd.query_devices(sd.default.device, 'input')
    samplerate = int(device_info['default_samplerate'])
    channels = 2 
    audio_data = []
    #recording the audio.
    i = 0
    while i < 10:
            audio_chunk = sd.rec(int(samplerate), samplerate=samplerate, channels=channels)
            sd.wait()
            audio_data.extend(audio_chunk)
            i += 1
    audio_data = np.array(audio_data)
    if len(audio_data) > 0:
        output_filename = "recorded_audio.flac"
        sf.write(output_filename, audio_data, samplerate)
        # st.write(f"Audio saved as {output_filename}")
    else:
        st.write("No audio recorded.")
    
def calculations(operation, operands, keyword, document):
    for token in document:
        if token.text.lower() in keyword :
            operation.append(token.text.lower())
        elif token.pos_ == "NUM":
            operands.append(int(token.text))
        result = None
    if all(word in [token.text for token in document] for word in ["board", "mass"]):    
        if len(operation)>0 and len(operands)>=2:
            result = str(operands[0])
            operation_index = 0
            for i in range(1, len(operands)):
                next_token = document[i+1]
                if operation[operation_index] in ["add", "plus", "edition", "sum", "addition", "adding", "summarize", "summarizing", "summation"]:
                    result += '+' + str(operands[i])
                elif operation[operation_index] in ["minus", "subtract"]:
                    result += '-' + str(operands[i])
                elif operation[operation_index] in ["multiply", "multiplication", "multiplying", "multiplied", "into"]:
                    result += '*' + str(operands[i])
                elif operation[operation_index] in ["divide", "division", "divided"]:
                    result += '/' + str(operands[i])
                operation_index += 1
                if operation_index >= len(operation):
                    break
        else:
            print("Failed to recognize a valid mathematical expression.")
        print(result)
        return eval(result)
    else:
        if len(operation)>0 and len(operands)>=2:
            result = operands[0]
            operation_index = 0
            for i in range(1, len(operands)):
                if operation[operation_index] in ["add", "plus", "edition", "sum", "addition", "adding", "summarize", "summarizing", "summation"]:
                    result += operands[i]
                elif operation[operation_index] == "minus":
                    result -= operands[i]
                elif operation[operation_index] == "subtract":
                    result = operands[i] - result
                elif operation[operation_index] in ["multiply", "multiplication", "multiplying", "multiplied", "into"]:
                    result *= operands[i] 
                elif operation[operation_index] in ["divide", "division", "divided"]:
                    result = (result/operands[i])
                operation_index += 1
                if operation_index >= len(operation):
                    break
        else:
            print("Failed to recognize a valid mathematical expression.")
        return result
       

def main():
    devices = sd.query_devices()
    print(devices)

    # Add custom CSS to the Streamlit app
    st.markdown(
        """
        <style>
        body {
            background-color: black;
            color: white;
            font-family: Arial, sans-serif;
            padding: 20px;
        }

        h1 {
            text-align: center;
        }

        p {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Streamlit content
    st.title("___VoCal___")
    st.write("VoCal your personal VoiceCalculator")

    if st.button("Click to record", type="primary"):
        record_audio()
        with st.spinner("Calculating..."):
            output = query("recorded_audio.flac")['text']
            main_o = output
            print(main_o)
            # Load the English language model
            import spacy
            nlp = spacy.load("en_core_web_sm")                  
            # Parse the recognized text using spaCy
            doc = nlp(main_o)
            operation=[]
            operands=[]
            all_operations = ["add", "plus", "sum", "edition", "addition", "adding", "summarize", "summarizing", "summation", "minus", "subtract", "multiply", "multiplication", "divide", "multiplying", "division", "divided", "multiplied", "into"]
            result = calculations(operands=operands, operation=operation, keyword=all_operations, document=doc)
            result = f"{result:.2f}"
            # Convert the result to speech
            engine = pyttsx3.init()
        if result is not None:
            engine.setProperty("rate", 120)
            engine.say(f"The result the operation is {result}")
            st.write(f"The result of the operation is {result}")
            engine.runAndWait()
        else:
            st.write("Invalid input. Please try again.")

if __name__ == "__main__":
    main()




