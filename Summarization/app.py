import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Sidebar for API key input
st.sidebar.title("Settings")
hf_api_key = st.sidebar.text_input("Enter your Hugging Face API Key", type="password")

# Model selection
st.title("Abstractive Text Summarization")
st.write("Select a model and enter text below to generate a summary.")

model_choice = st.selectbox("Choose a model:", ["k200353/t5-small-finetuned-cnn-dailymail" ,"facebook/bart-large-cnn", "t5-small","t5-base", "google/pegasus-xsum", "k200353/pegasus-finetuned-cnn_dailymail"])

# Load the selected model with API key
if hf_api_key:
    tokenizer = AutoTokenizer.from_pretrained(model_choice, token=hf_api_key)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_choice, token=hf_api_key)
else:
    st.sidebar.warning("Please enter your Hugging Face API key to load the model.")
    summarizer = None

# Example texts
examples = {
    "Climate": "Climate change refers to long-term shifts in temperatures and weather patterns, primarily caused by human activities such as burning fossil fuels. These activities increase greenhouse gas levels in the atmosphere, trapping heat and leading to global warming. The effects of climate change include rising sea levels, more extreme weather events, and loss of biodiversity. Scientists emphasize the urgent need for reducing carbon emissions and transitioning to renewable energy sources to mitigate these effects and protect the planet for future generations.",
    "Apollo": "In 1969, NASA successfully landed the first humans on the Moon as part of the Apollo 11 mission. Astronauts Neil Armstrong and Buzz Aldrin spent about 21 hours on the lunar surface, collecting samples and conducting experiments. This historic event marked a significant milestone in space exploration, inspiring generations of scientists and engineers. The Apollo program continued with subsequent missions, expanding our understanding of the Moon and paving the way for future space travel. Today, agencies like NASA and private companies are working towards returning humans to the Moon through the Artemis program, with the goal of establishing a sustainable presence and preparing for future missions to Mars.",
    "Aliens": """Some people belive that the so called "face" on mars was created by life on mars. This is not the case. The face on Mars is a naturally occuring land form called a mesa. It was not created by aliens, and there is no consiracy to hide alien lifeforms on mars. There is no evidence that NASA has found that even suggests that this face was created by aliens.\n\nA mesa is a naturally occuring rock formation, that is found on Mars and Earth. This "face" on mars only looks like a face because humans tend to see faces wherever we look, humans are obviously extremely social, which is why our brain is designed to recognize faces.\n\nMany conspiracy theorists believe that NASA is hiding life on Mars from the rest of the world. These people would be very wrong. If NASA found life on Mars, then they would get millions of people\'s attention. NASA\'s budget would increase drasticly, which means that their workers would get paid more. There is no good reason that NASA would hide life on Mars from the rest of the world.\n\nSo, NASA is not hiding life on Mars from us, and they are not trying to trick us into thinking that the "face" on mars is just a mesa, because it actually is. NASA hiding life would be illogical, because if they found life on Mars, they would make a lot of money, and we all know that the people at NASA aren\'t illogical people."""
}

# Text input area

# Initialize session state if not already set
if "text_input" not in st.session_state:
    st.session_state.text_input = ""

# Display example buttons
st.write("Select an example:")
for key, value in examples.items():
    if st.button(value):
        st.session_state.text_input = value  # Overwrite text area with selected example

input_text = st.text_area("Enter your text:", height=200, key="text_input")

if st.button("Summarize"):
    if model and input_text:
        with st.spinner("Generating summary..."):
            tokens = tokenizer(input_text, truncation=True, padding="longest", return_tensors="pt")
            summary = model.generate(**tokens)

            if model_choice == "t5-small" or model_choice == "t5-base":
                st.write("For:", model_choice)
                tokens=tokenizer.encode("sumarize: " +input_text,return_tensors='pt', padding="longest", truncation=True)
                summary = model.generate(tokens)
            
            if model_choice == "k200353/t5-small-finetuned-cnn-dailymail":
                #These sets of hyper parameters helps to give the best summarization
                inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="longest")
                summary = model.generate(inputs['input_ids'], max_length=150, num_beams=4, early_stopping=True)

            decoded_output = tokenizer.decode(summary[0], skip_special_tokens=True)


            st.subheader("Generated Summary")
            st.write(decoded_output)
    elif not summarizer:
        st.warning("Model not loaded. Please enter a valid Hugging Face API key.")
    else:
        st.warning("Please enter text before summarizing.")
