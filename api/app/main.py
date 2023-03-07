"""Main module to run the demo app."""
from io import BytesIO

from PIL import Image
import streamlit as st
from streamlit_image_comparison import image_comparison
from super_resolve import get_prediction


if __name__ == "__main__":
    st.header("Super resolution.")

    uploaded_file = st.file_uploader("Choose an image")

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image.filename = uploaded_file.name
        st.image(image, caption="Original image", use_column_width="always")

        if st.button("Process image"):
            sr_image = get_prediction(image)
            st.image(
                sr_image, caption="Super resolution image", use_column_width="always"
            )

            image_comparison(
                img1=image.convert("RGB"),
                img2=sr_image.convert("RGB"),
                label1="Original image",
                label2="Super resolution image",
                width=700,
                starting_position=50,
                show_labels=True,
                make_responsive=True,
                in_memory=True,
            )

            buf = BytesIO()
            sr_image.save(buf, format="PNG")
            byte_image = buf.getvalue()

            download_button = st.download_button(
                label="Download image",
                data=byte_image,
                file_name=f"{uploaded_file.name}",
                mime="image/png",
            )
