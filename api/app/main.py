"""Main module to run the demo app."""
from pathlib import Path

from PIL import Image
import streamlit as st
from streamlit_image_comparison import image_comparison
from super_resolve import get_prediction


if __name__ == "__main__":
    st.header("Super resolution.")

    uploaded_file = st.file_uploader("Choose an image")

    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        input_image.filename = uploaded_file.name
        st.image(input_image, caption="Original image", use_column_width="always")

        if st.button("Process image"):
            output_image_path = get_prediction(input_image)  # Predicted as "jpg".
            output_image = Image.open(output_image_path).convert("RGB")

            st.image(
                output_image,
                caption="Super resolution image",
                use_column_width="always",
            )

            image_comparison(
                img1=input_image,
                img2=output_image,
                label1="Original image",
                label2="Super resolution image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )

            with open(output_image_path, "rb") as f:
                image_bytes = f.read()

            download_button = st.download_button(
                label="Download image",
                data=image_bytes,
                file_name=f"{Path(uploaded_file.name).stem}.jpg",
                mime="image/jpeg",
            )
