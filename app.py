import streamlit as st
from src.session_states import initialize_params

# Set the streamlit page parameters
st.set_page_config(layout="wide",
                   initial_sidebar_state='expanded',
                   page_icon="👻",
                   page_title='GPT Document Analyzer')

##############################################################################################################
############################# Function for displaying the introduction page ##################################
##############################################################################################################
def display_intro():
    #################################### section 1 => text / Information #####################################
    col_left, col_right = st.columns(2)

    with col_left:
        st.title(":rainbow[TOYOTA USER MANUEL]")
        st.write('')
        st.subheader(":grey[Got questions about the Toyota Highlander?]")
        st.write(":grey[Interact with our chatbot by using the side menu...]")

    with col_right:
        image_path = "https://raw.githubusercontent.com/Samuelchazy/Educative.io/0d5526803b4f9993b069d3c4460fe3caf69e553e/images/toyota_logo.png"
        # print(st.image_path)
        st.image(image_path, use_container_width=True)

    ###################################### section 2 => Image & Video #######################################
    col_left, col_right = st.columns([0.45, 0.55])

    with col_left:
        video_path = "https://www.youtube.com/watch?v=N-EQ_08Ptu4"
        st.video(video_path)

    with col_right:
        image_path = "https://raw.githubusercontent.com/Samuelchazy/Educative.io/f3a444c05570820c63ba9b79c68eb0f2b1214b92/images/Toyota_1.jpg"
        st.image(image_path, use_container_width=True)

initialize_params()
display_intro()
