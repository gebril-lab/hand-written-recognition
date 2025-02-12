import streamlit as st


from streamlit_drawable_canvas import st_canvas

import numpy as np
import random

import onnxruntime as ort

import cv2

from PIL import Image

from letter_detector import detect_letters, merge_nearby_boxes, pad_and_center_image
import time

class HandwrittenLetterRecognition:

    def __init__(self):
        self.canvas_size = (800, 400)
        self.onnx_session = ort.InferenceSession('emnist.onnx')

    def setup_ui(self):
        st.title("Handwritten Letter Recognition")

        # Canvas settings
        stroke_width = st.slider("Stroke width: ", 1, 50, 20)

        # Initialize session state
        if 'target_letters' not in st.session_state:
            st.session_state.target_letters = [chr(random.randint(ord('a'), ord('z'))) for _ in range(5)]

        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0

        if 'score' not in st.session_state:
            st.session_state.score = 0

        if 'trial_count' not in st.session_state:
            st.session_state.trial_count = 0

            # Display progress bar
        progress = (st.session_state.current_index / 5) 
        st.progress(progress)
        st.write(f"Progress: {st.session_state.current_index}/5 letters")


        if st.session_state.current_index < 5:
            current_letter = st.session_state.target_letters[st.session_state.current_index]
            st.subheader(f"Please write the letter: {current_letter.upper()} ")
        else:
            st.balloons()
            st.success(f"🎉Test complete! Your score: {st.session_state.score}/5")
            
            if st.button("Restart Test"):
                self.reset_test()
            return

        # Handle canvas clearing
        if 'canvas_key' not in st.session_state:
            st.session_state.canvas_key = 0

        # Create the canvas
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=stroke_width,
            stroke_color="black",
            background_color="white",
            height=400,
            width=800,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state.canvas_key}",
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Predict"):
                if canvas_result.image_data is not None:
                    self.predict_letter(canvas_result.image_data)

        with col2:
            if st.button("Clear"):
                st.session_state.canvas_key += 1
                st.rerun()

        with col3:
            if st.button("Next Letter"):
                self.next_letter()

        # Display preview area
        self.preview = st.empty()

    def reset_test(self):
        st.session_state.target_letters = [chr(random.randint(ord('a'), ord('z'))) for _ in range(5)]
        st.session_state.current_index = 0
        st.session_state.score = 0
        st.session_state.trial_count = 0
        st.session_state.canvas_key += 1
        st.rerun()

    def next_letter(self):
        st.session_state.current_index += 1
        st.session_state.trial_count = 0
        st.session_state.canvas_key += 1
        st.rerun()

    def predict_letter(self, canvas_data):
        # Convert canvas data to grayscale image
        img_gray = cv2.cvtColor(canvas_data, cv2.COLOR_RGBA2GRAY)

        # Detect and process letters
        boxes = detect_letters(img_gray)
        merged_boxes = merge_nearby_boxes(boxes)

        if not merged_boxes:
            st.warning("No letters detected. Please draw something.")
            return

        if len(merged_boxes) > 1:
            st.error("Multiple letters detected. Please write only one letter.")
            return

        # Prepare the single letter image
        x1, y1, x2, y2 = merged_boxes[0]
        letter_img = img_gray[y1:y2, x1:x2]
        padded_img = pad_and_center_image(letter_img)
        resized_img = cv2.resize(padded_img, (28, 28), interpolation=cv2.INTER_NEAREST)
        normalized_img = resized_img.astype(np.float32) / 255.0

        # Add batch and channel dimensions
        letter_image_batch = np.expand_dims(normalized_img, axis=(0, 1))  # Shape: (1, 1, 28, 28)

        # Run prediction
        input_name = self.onnx_session.get_inputs()[0].name
        result = self.onnx_session.run(None, {input_name: letter_image_batch})

        # Map predictions to characters
        index_to_char_map = {
            i: chr(ord('a') + i - 1) if i > 0 else 'N/A'
            for i in range(27)
        }

        prediction = index_to_char_map.get(np.argmax(result[0]), '?')

        # Check prediction against target letter
        current_letter = st.session_state.target_letters[st.session_state.current_index]
        if prediction == current_letter:
            st.success(f"✅ Correct! ")
            st.session_state.score += 1
            time.sleep(1) 
            self.next_letter()
        else:
            st.session_state.trial_count += 1
            if st.session_state.trial_count >= 3:
                st.error(f" ❌ you failed in this letter .\n click next letter, to complete your test ")
                # time.sleep(5) 
                # self.next_letter()
            else:
                st.error(f" ❌ Incorrect, Keep trying! \n {3-st.session_state.trial_count} trials left .")




def main():
    st.set_page_config(page_title="Handwritten Letter Recognition")
    app = HandwrittenLetterRecognition()
    app.setup_ui()


if __name__ == "__main__":
    main()
