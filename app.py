"""Local QC Inspector — an offline Streamlit GUI for 10-point presence checking."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from qc_engine import (
    add_good_samples,
    add_missing_samples,
    create_profile,
    draw_layout_preview,
    image_to_png_bytes,
    inspect_image,
    list_profiles,
    load_profile,
    profile_dir,
    profile_summary,
    read_image_bytes,
    reset_profile,
    update_layout,
)


APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = APP_DIR / "qc_data"
DATA_ROOT.mkdir(exist_ok=True)

st.set_page_config(page_title="Local QC Inspector", page_icon="✅", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1450px; padding-top: 1.6rem; }
      .qc-pass { background:#e8f8ed; border:2px solid #138a36; border-radius:10px; padding:18px 22px; font-size:1.25rem; color:#075c22; }
      .qc-fail { background:#ffebeb; border:2px solid #ca2525; border-radius:10px; padding:18px 22px; font-size:1.25rem; color:#8d0909; }
      .qc-review { background:#fff6df; border:2px solid #e69b00; border-radius:10px; padding:18px 22px; font-size:1.25rem; color:#785000; }
      .muted { color:#586069; }
    </style>
    """,
    unsafe_allow_html=True,
)


def uploaded_pairs(files: list[Any] | None) -> list[tuple[str, bytes]]:
    return [(file.name, file.getvalue()) for file in (files or [])]


def render_profile_creator() -> None:
    st.title("Local QC Inspector")
    st.subheader("Create your first inspection profile")
    st.write(
        "This app runs entirely on this computer. Start by adding pictures of a fully assembled, known-good part. "
        "The supplied photos are suitable for this first step."
    )
    st.info(
        "Use at least 10 good images for a usable setup; 20–30 images from the actual factory camera and lighting are better. "
        "The app includes default locations for the 8 top screws and 2 bottom blue clips, which you can verify in the Layout tab."
    )
    with st.form("new_profile"):
        name = st.text_input("Profile name", value="Seat blower assembly")
        files = st.file_uploader(
            "Known-good part images",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            accept_multiple_files=True,
            help="Choose only images where all 8 screws and both blue clips are installed.",
        )
        create = st.form_submit_button("Create profile and calibrate", type="primary")
    if create:
        try:
            profile_name = create_profile(DATA_ROOT, name, uploaded_pairs(files))
            st.session_state["profile"] = profile_name
            st.success(f"Created and calibrated '{profile_name}'.")
            st.rerun()
        except Exception as error:
            st.error(str(error))
    st.markdown("---")
    st.caption("Images are saved only under this app's qc_data folder. The app does not use an internet service or cloud model.")


def outcome_panel(report: dict[str, Any]) -> None:
    outcome = report["outcome"]
    if outcome == "PASS":
        st.markdown('<div class="qc-pass"><b>PASS — nothing is missing.</b><br>All 10 inspection locations match this profile.</div>', unsafe_allow_html=True)
    elif outcome == "FAIL":
        st.markdown(
            f'<div class="qc-fail"><b>FAIL — missing: {", ".join(report["missing"])}.</b><br>Do not pass this part until the missing items are verified.</div>',
            unsafe_allow_html=True,
        )
    else:
        detail = ", ".join(report["review"]) if report["review"] else "camera alignment"
        st.markdown(
            f'<div class="qc-review"><b>REVIEW REQUIRED.</b><br>Unable to make a reliable automatic decision for: {detail}.</div>',
            unsafe_allow_html=True,
        )


def render_inspection(profile_name: str, profile: dict[str, Any]) -> None:
    st.header("Inspect a part")
    st.write("Upload a photo or capture one from the connected camera. The result is calculated locally.")
    source_mode = st.radio("Image source", ["Upload image", "Connected camera"], horizontal=True)
    if source_mode == "Upload image":
        source = st.file_uploader("Part image", type=["jpg", "jpeg", "png", "bmp", "webp"], key="inspection_upload")
    else:
        source = st.camera_input("Take a picture of the part", key="inspection_camera")
        st.caption("Allow browser camera access when asked. Keep the camera facing the part in the same orientation as the profile reference.")

    if source is None:
        st.info("Select or capture a part image to begin inspection.")
        return
    image_bytes = source.getvalue()
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.image(image_bytes, caption="Input image", use_container_width=True)
        run = st.button("Run 10-point inspection", type="primary", use_container_width=True)
    with right:
        st.markdown("#### Inspection scope")
        st.write("- Screw 1 through Screw 8 (the eight silver fasteners on the upper metallic plate)")
        st.write("- Clip 1 and Clip 2 (the two blue clips at the bottom)")
        summary = profile_summary(profile)
        st.caption(f"Current profile: {summary['good_images']} good image(s), {summary['missing_images']} labelled defect image(s).")

    if run:
        with st.spinner("Registering the image and checking each location…"):
            try:
                report = inspect_image(DATA_ROOT, profile_name, image_bytes)
            except Exception as error:
                st.error(f"Inspection could not run: {error}")
                return
        st.markdown("---")
        outcome_panel(report)
        st.caption(
            f"Image alignment: {report['alignment']['method']} · "
            f"{report['alignment']['inliers']} reliable feature matches. "
            "Use REVIEW rather than PASS if the fixture/camera alignment is not stable."
        )
        display, table = st.columns([1.25, 1], gap="large")
        with display:
            st.image(image_to_png_bytes(report["annotated_image"]), caption="Inspection overlay (green = present, red = missing, amber = review)", use_container_width=True)
            st.download_button(
                "Download annotated result (.png)",
                data=image_to_png_bytes(report["annotated_image"]),
                file_name="qc_inspection_result.png",
                mime="image/png",
            )
        with table:
            st.markdown("#### Location results")
            rows = []
            for result in report["results"]:
                rows.append(
                    {
                        "ID": result["id"],
                        "Type": result["type"].title(),
                        "Result": result["status"],
                        "Confidence": f"{result.get('confidence', 0)}%",
                        "Explanation": result["message"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=460)
            st.caption("Confidence is a similarity score, not a safety certification. Validate the threshold and defect examples before using the result as a production release decision.")


def render_training(profile_name: str, profile: dict[str, Any]) -> None:
    st.header("Training & calibration")
    st.write("Use this page whenever the camera, mount, part presentation, or lighting changes. No coding is needed.")
    summary = profile_summary(profile)
    one, two, three = st.columns(3)
    one.metric("Good images", summary["good_images"])
    two.metric("Labelled defect images", summary["missing_images"])
    three.metric("Locations with defect examples", sum(1 for value in summary["defect_counts"].values() if value > 0))

    st.subheader("1. Add fully assembled (good) parts")
    st.caption("Add images where every screw and clip is present. Include normal production variation: small position shifts, brightness changes, and allowed part variation.")
    good_files = st.file_uploader(
        "Good part images to add",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="good_training_files",
    )
    if st.button("Add good images and retrain", type="primary", disabled=not good_files, key="add_good"):
        try:
            count = add_good_samples(DATA_ROOT, profile_name, uploaded_pairs(good_files))
            st.success(f"Added {count} good image(s) and retrained the local model.")
            st.rerun()
        except Exception as error:
            st.error(str(error))

    st.markdown("---")
    st.subheader("2. Add examples with a known missing item")
    st.warning(
        "For reliable automatic defect calls, add real images for each missing condition you need to detect. "
        "For example, select “Screw 1” and upload only images where Screw 1 is actually missing."
    )
    defect_files = st.file_uploader(
        "Defect images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="defect_training_files",
    )
    selected_missing = st.multiselect(
        "Items missing in every selected image",
        options=[item["id"] for item in profile["items"]],
        help="If the selected images have different missing items, add them in separate batches.",
    )
    if st.button("Add labelled defect images and retrain", type="primary", disabled=not defect_files or not selected_missing, key="add_defect"):
        try:
            count = add_missing_samples(DATA_ROOT, profile_name, uploaded_pairs(defect_files), selected_missing)
            st.success(f"Added {count} labelled defect image(s) and retrained the local model.")
            st.rerun()
        except Exception as error:
            st.error(str(error))

    st.markdown("---")
    st.subheader("Training coverage")
    coverage_rows = []
    for item in profile["items"]:
        count = summary["defect_counts"].get(item["id"], 0)
        coverage_rows.append(
            {
                "ID": item["id"],
                "Type": item["kind"].title(),
                "Labelled missing images": count,
                "Ready for strongest decision": "Yes" if count >= 3 else "No — add 3+ real examples",
            }
        )
    st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, use_container_width=True)
    if summary["warnings"]:
        st.warning("Some saved samples could not be read during the last calibration: " + ", ".join(summary["warnings"]))


def render_layout(profile_name: str, profile: dict[str, Any]) -> None:
    st.header("Layout: verify the 10 inspection locations")
    st.write(
        "The orange boxes must sit over the center of each screw/clip in the reference image. "
        "Default positions were set for the supplied part. Change the normalized numbers below if your view is different."
    )
    root = profile_dir(DATA_ROOT, profile_name)
    try:
        reference = read_image_bytes((root / profile["reference_image"]).read_bytes())
        st.image(image_to_png_bytes(draw_layout_preview(reference, profile["items"])), caption="Reference image with current 10 inspection boxes", use_container_width=True)
    except Exception as error:
        st.error(f"Could not display the profile reference: {error}")
        return
    st.info("Coordinates are percentages of the reference image: 0.50 is the centre, 0.10 is 10% from the left/top. Width and height are box sizes as percentages.")
    table = pd.DataFrame(profile["items"])[["id", "kind", "center_x", "center_y", "width", "height"]]
    edited = st.data_editor(
        table,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "id": st.column_config.TextColumn("Item ID", required=True),
            "kind": st.column_config.SelectboxColumn("Type", options=["screw", "clip"], required=True),
            "center_x": st.column_config.NumberColumn("Center X", min_value=0.001, max_value=0.999, step=0.001, format="%.3f"),
            "center_y": st.column_config.NumberColumn("Center Y", min_value=0.001, max_value=0.999, step=0.001, format="%.3f"),
            "width": st.column_config.NumberColumn("Width", min_value=0.005, max_value=0.2, step=0.001, format="%.3f"),
            "height": st.column_config.NumberColumn("Height", min_value=0.005, max_value=0.2, step=0.001, format="%.3f"),
        },
        key="layout_editor",
    )
    if st.button("Save layout and recalibrate", type="primary"):
        try:
            update_layout(DATA_ROOT, profile_name, edited.to_dict(orient="records"))
            st.success("Layout saved and all local models recalibrated.")
            st.rerun()
        except Exception as error:
            st.error(str(error))


def render_guide() -> None:
    st.header("Operating guide")
    st.markdown(
        """
        ### Daily use

        1. Place the part in its fixture with the upper metal plate at the top of the camera view.
        2. Use the same camera height, focus, and lighting used for training.
        3. In **Inspect a part**, capture/upload an image and press **Run 10-point inspection**.
        4. PASS means all ten locations match training. FAIL names each missing location. REVIEW means an operator must look at the part or collect more training images.

        ### When setting up a new camera or station

        1. Create a separate profile for that station.
        2. Add 20–30 known-good parts from the actual station, including normal brightness and placement variation.
        3. Open **Layout** and make sure the ten orange boxes are centered on their locations.
        4. For every defect type used in production, add at least 3–5 labelled photos in **Training & calibration**.
        5. Challenge the station with a documented test set of good and bad parts before making automatic production decisions. Record false passes/false fails and retrain when needed.

        ### Important limits

        This application checks visual presence at fixed locations. It does not prove torque, correct screw type, hidden assembly state, or electrical function. A fixed fixture, diffuse lighting, and a camera that is locked in position make the inspection much more repeatable.
        """
    )


def main() -> None:
    profiles = list_profiles(DATA_ROOT)
    if not profiles:
        render_profile_creator()
        return

    with st.sidebar:
        st.title("✅ Local QC Inspector")
        chosen = st.selectbox("Inspection profile", profiles, key="selected_profile")
        st.caption("All training images and models are stored locally.")
        with st.expander("Profile actions"):
            st.write("Create a fresh profile for a different part, camera, or fixture.")
            with st.form("additional_profile"):
                additional_name = st.text_input("New profile name")
                additional_files = st.file_uploader("Known-good images", type=["jpg", "jpeg", "png", "bmp", "webp"], accept_multiple_files=True, key="additional_profile_files")
                make_additional = st.form_submit_button("Create new profile")
            if make_additional:
                try:
                    new_name = create_profile(DATA_ROOT, additional_name, uploaded_pairs(additional_files))
                    st.success(f"Created '{new_name}'.")
                    st.rerun()
                except Exception as error:
                    st.error(str(error))
            st.divider()
            confirm = st.checkbox("I understand this permanently deletes this profile's training data", key="delete_confirmation")
            if st.button("Delete selected profile", disabled=not confirm, type="secondary"):
                reset_profile(DATA_ROOT, chosen)
                st.rerun()

    profile = load_profile(DATA_ROOT, chosen)
    st.title(f"{profile['name']} · 10-point presence inspection")
    inspection_tab, training_tab, layout_tab, guide_tab = st.tabs(["Inspect a part", "Training & calibration", "Layout", "Operating guide"])
    with inspection_tab:
        render_inspection(chosen, profile)
    with training_tab:
        render_training(chosen, profile)
    with layout_tab:
        render_layout(chosen, profile)
    with guide_tab:
        render_guide()


if __name__ == "__main__":
    main()
