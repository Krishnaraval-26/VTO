# IMPORTS

import numpy as np
import os, io, json, base64, boto3
import streamlit as st
from PIL import Image, ImageOps
from botocore.config import Config

# CONFIG 
REGION   = os.getenv("AWS_REGION", "us-east-1")
MODEL_ID = "amazon.nova-canvas-v1:0"

# HELPERS 
def normalize_image(file, force_format: str):
    """
    Accepts a JPG/PNG file-like object.
    - Fix EXIF orientation
    - Keep aspect ratio
    - Enforce each side in [320, 4096] (downscale if too big, upscale if tiny)
    - Re-encode to force_format (JPEG or PNG)
    Returns: base64 string
    """
    img = Image.open(file)
    img = ImageOps.exif_transpose(img)

    # Size constraints per Nova 
    min_side, abs_cap = 320, 4096
    w, h = img.size

    # Cap long side <= 3072 (safe quality, under 4096 hard limit)
    target_max = 3072
    long_side = max(w, h)
    if long_side > target_max:
        scale = target_max / float(long_side)
        img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        w, h = img.size

    # Ensure short side >= 320 (rare, but makes tiny inputs valid)
    short_side = min(w, h)
    if short_side < min_side:
        scale = float(min_side) / float(short_side)
        new_w, new_h = int(w*scale), int(h*scale)
        # final safety: never exceed 4096
        if max(new_w, new_h) > abs_cap:
            s2 = abs_cap / float(max(new_w, new_h))
            new_w, new_h = int(new_w*s2), int(new_h*s2)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    # Encode
    buf = io.BytesIO()
    fmt = force_format.upper()
    if fmt == "JPEG" and img.mode != "RGB":
        img = img.convert("RGB")
    if fmt == "JPEG":
        img.save(buf, "JPEG", quality=92, optimize=True)
    else:
        img.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def invoke_vto(source_b64, reference_b64, garment_class, width, height, cfg, seed):
    payload = {
        "taskType": "VIRTUAL_TRY_ON",
        "virtualTryOnParams": {
            "sourceImage": source_b64,
            "referenceImage": reference_b64,
            "maskType": "GARMENT",
            "garmentBasedMask": {"garmentClass": garment_class},
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,
            "height": int(height),
            "width": int(width),
            "cfgScale": float(cfg),
            "seed": int(seed),
        },
    }
    brt = boto3.client("bedrock-runtime", region_name=REGION, config=Config(read_timeout=300))
    resp = brt.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps(payload),
        accept="application/json",
        contentType="application/json",
    )
    body = json.loads(resp["body"].read())
    return body["images"][0]  # base64 string


# STREAMLIT UI


st.set_page_config(page_title="Try On", layout="centered")
st.title("Virtual Try-On")

# --- helper: placeholder image for alignment (render ONLY inside result_box) ---
PLACEHOLDER_PX = 420
def placeholder_img():
    return np.zeros((PLACEHOLDER_PX, PLACEHOLDER_PX, 3), dtype=np.uint8) + 30

# --- state init ---
if "person_bytes" not in st.session_state:   st.session_state.person_bytes  = None
if "product_bytes" not in st.session_state:  st.session_state.product_bytes = None
if "vto_result_bytes" not in st.session_state: st.session_state.vto_result_bytes = None

# --- 3 columns layout: Person | Product | Result ---
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Person Image**")
    st.caption("Please upload a JPG, JPEG, or PNG image.")
    person_slot = st.empty()  # uploader OR preview
    if st.session_state.person_bytes is None:
        _pf = person_slot.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="person_upload",
        )
        if _pf:
            st.session_state.person_bytes = _pf.getvalue()
            st.session_state.person_name  = _pf.name
            st.rerun()
    else:
        st.image(st.session_state.person_bytes, use_container_width=True, caption="Person preview")

with c2:
    st.markdown("**Product Image**")
    st.caption("Please upload a JPG, JPEG, or PNG image.")
    product_slot = st.empty()
    if st.session_state.product_bytes is None:
        _gf = product_slot.file_uploader(
            "",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            key="garment_upload",
        )
        if _gf:
            st.session_state.product_bytes = _gf.getvalue()
            st.session_state.product_name  = _gf.name
            st.rerun()
    else:
        st.image(st.session_state.product_bytes, use_container_width=True, caption="Product preview")

with c3:
    st.markdown("**Generated Image**")
    st.caption("Wait while the image gets generated, the image will pop here")
    result_box = st.empty()  # ONLY place where we render placeholder/result
    if st.session_state.person_bytes and st.session_state.product_bytes:
        if st.session_state.vto_result_bytes:
            result_box.image(st.session_state.vto_result_bytes, use_container_width=True)
        else:
            result_box.image(placeholder_img(), use_container_width=True)
    else:
        st.session_state.vto_result_bytes = None
        result_box.image(placeholder_img(), use_container_width=True)

# --- one row of buttons under the three images ---
b1, b2, b3 = st.columns(3)
with b1:
    if st.session_state.person_bytes is not None:
        if st.button("Change", key="change_person_row"):
            st.session_state.person_bytes = None
            st.session_state.vto_result_bytes = None
            st.rerun()
with b2:
    if st.session_state.product_bytes is not None:
        if st.button("Change", key="change_product_row"):
            st.session_state.product_bytes = None
            st.session_state.vto_result_bytes = None
            st.rerun()
with b3:
    if st.session_state.vto_result_bytes:
        st.download_button(
            "Download PNG",
            data=st.session_state.vto_result_bytes,
            file_name=st.session_state.get("vto_result_name", "vto_result.png"),
            mime="image/png",
            key="download_row",
        )
    else:
        st.write("")

# --- controls ---
garment_class = st.radio(
    "Garment class",
    options=["FULL_BODY", "UPPER_BODY", "LOWER_BODY"],
    horizontal=True,
    # index=0,
    key="garment_class"
)

with st.expander("Advanced", expanded=False):
    width  = st.slider("Output width",  768, 1536, 1024, step=64)
    height = st.slider("Output height", 768, 1536, 1024, step=64)
    cfg    = st.slider("CFG scale", 1.0, 15.0, 8.0, step=0.5)
    seed   = st.number_input("Seed", value=0, min_value=0, max_value=10_000)

# --- convert session bytes -> file-like objects for helpers ---
person_file  = io.BytesIO(st.session_state.person_bytes)  if st.session_state.person_bytes  else None
product_file = io.BytesIO(st.session_state.product_bytes) if st.session_state.product_bytes else None

run = st.button("Generate Try-On", type="primary", disabled=not(person_file and product_file))

if run:
    try:
        with st.spinner("Encoding & resizing images…"):
            src_b64 = normalize_image(person_file,  force_format="JPEG")
            ref_b64 = normalize_image(product_file, force_format="PNG")

        with st.spinner("Trying New Product…"):
            out_b64 = invoke_vto(src_b64, ref_b64, garment_class, width, height, cfg, seed)

        out_bytes = base64.b64decode(out_b64)
        st.success("Done.")

        st.session_state.vto_result_bytes = out_bytes
        st.session_state.vto_result_name  = f"vto_{garment_class.lower()}_{seed}.png"

        # update the right column (replaces placeholder)
        result_box.image(out_bytes, use_container_width=True)

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.stop()

st.markdown("""
**Tips**
- Use **UPPER_BODY** / **LOWER_BODY** / **FULL_BODY** as needed.
- Oversized/undersized images are auto-normalized before calling Nova.
- Keep product images tightly cropped for cleaner results.
""")
