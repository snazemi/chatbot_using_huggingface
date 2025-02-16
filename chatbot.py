# Step 2: (after installing pre-requisits) Import the necessary libraries
# Class AutoModelForSeq2SeqLM is imported, for a 'model' instance, that allows you to interact with your chosen language model.
# Class AutoTokenizer is imported, for a 'tokenizer' instance, which optimizes your input and passes it to the language model efficiently. It does so by converting your text input to “tokens”, which is how the model interprets the text.
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Step 3: select the model name (use https://huggingface.co/models for any other mode you'd like to use)
# For this example, you'll be using facebook/blenderbot-400M-distill because it has an open-source license and runs relatively fast.
model_name = "facebook/blenderbot-400M-distill"

# Step 4: Fetch the model and initialize a tokenizer
# Load model (download on first run and reference local installation for consequent runs)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 5: Chat

# Step 5.1: Keeping track of conversation history
conversation_history = []

# Start a loop
while True:

    # Step 5.2: Encoding the conversation history
    history_string = "\n".join(conversation_history)

    # Step 5.3: Fetch prompt from user
    input_text = input("Prompt > ")

    # Step 5.4: Tokenization of user prompt and chat history
    inputs = tokenizer.encode_plus(history_string, input_text, return_tensors="pt")
    # print(inputs)

    # Step 5.5: Generate output from the model
    outputs = model.generate(**inputs)
    #print(outputs)

    # Step 5.6: Decode output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print(response)

    # Step 5.7: Update conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)
    #print(conversation_history)