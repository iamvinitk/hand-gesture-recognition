import openai, json, os
from gtts import gTTS
from playsound import playsound

workdir = os.path.abspath(os.path.dirname(__file__))
print(workdir)
with open(workdir + "/api_key.json", "r") as f:  # Please put your own API key into the API_Key.json first.
    keyDict = json.load(f)

MODEL = "gpt-3.5-turbo"

API_KEY = keyDict.get("personalTestKey")
openai.api_key = API_KEY

message_list = [
    {
        "role": "user",
        "content":
            '''
            1. peace: "Hey there!"
            2. thumbs_up: "All good, okay!"
            3. thumbs_down: "Not cool, not okay."
            4. fist: "I've got your back, everything will be okay."
            5. clarity: "Can you clarify or repeat that?"
            
            Whenever I say one of this inputs. You need to reply with a similar 1 sentence. Be ready to reply with 
            similar 1 line sentences.
            
            Example:
            thumbs_up - I'm doing good, thanks for asking!
            '''
    },
]


def get_response(model_id, message):
    """
    Test the model with a given message.

    :param model_id: ID of the model to test.
    :param message: Message to test the model.
    :return: Response from the model.
    """
    completion = openai.ChatCompletion.create(
        model=model_id,
        messages=message,
        temperature=0  # YZ 2023-11-1, set to 0 not default 1 for its small sample set
    )
    return completion.choices[0].message["content"]


def process_message(input_message):
    message_list.append({"role": "user", "content": input_message})
    response = get_response(MODEL, message_list)
    message_list.append({"role": "system", "content": response})
    tts = gTTS(response)
    tts.save("message.mp3")
    playsound("message.mp3")


def main():
    response = get_response(MODEL, message_list)
    message_list.append({"role": "system", "content": response})
    while True:
        user_input = input(response)
        message_list.append({"role": "user", "content": user_input})
        response = get_response(MODEL, message_list)
        message_list.append({"role": "system", "content": response})
        tts = gTTS(response)
        tts.save("message.mp3")
        playsound("message.mp3")


if __name__ == '__main__':
    main()
