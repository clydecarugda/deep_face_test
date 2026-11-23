
from deepface import DeepFace
import pandas as pd
from pathlib import Path

image_to_search = "/Volumes/T7/Clyde/Pictures/329066595_216437737539722_7095925692090534052_n.jpg"
input_folder = "/Volumes/T7/Clyde/Pictures"
output_folder = "output"

MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"
CONFIDENCE_THRESHOLD = 70


def find_similar_faces(image_to_search, input_folder):
    try:
        results = DeepFace.find(
            img_path=image_to_search,
            db_path=input_folder,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            detector_backend="opencv",
            enforce_detection=True,
            align=True,
            silent=False
        )

        return results

    except Exception as error:
        print(f"Face search failed: {error}")
        return None


def face_verification(image_to_verify, image_to_compare):
    try:
        result = DeepFace.verify(
            img1_path=image_to_verify,
            img2_path=image_to_compare,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            detector_backend="opencv",
            enforce_detection=True,
            align=True
        )

        print(f"Distance: {result['distance']}")
        print(f"Threshold: {result['threshold']}")
        print(f"Confidence: {result['confidence']}")

        return result["verified"]

    except Exception as error:
        print(f"Verification failed: {error}")
        return False


# Find similar faces
result = find_similar_faces(image_to_search, input_folder)

if result and not result[0].empty:

    df = result[0]

    # Filter results
    high_conf_df = df[
        df["confidence"] >= CONFIDENCE_THRESHOLD
    ].copy()

    Path(output_folder).mkdir(parents=True, exist_ok=True)

    high_conf_df.to_csv(
        f"{output_folder}/output.csv",
        index=False
    )

    print(f"\nMatches found: {len(high_conf_df)}")
    print(f"Results saved to {output_folder}/output.csv")

else:
    print("No similar faces found!")
