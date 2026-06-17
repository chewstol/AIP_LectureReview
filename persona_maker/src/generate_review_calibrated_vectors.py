import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REVIEW_MENTION_PATH = PROJECT_ROOT / "data" / "processed" / "review_mention_vectors.json"
LECTURE_VECTOR_PATH = PROJECT_ROOT / "data" / "csv" / "lecture_nodes_with_text.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "data" / "processed" / "review_calibrated_vectors.json"
OUTPUT_CSV_PATH = PROJECT_ROOT / "data" / "processed" / "review_calibrated_vectors.csv"

MENTION_FEATURES = [
    "assignment",
    "teamwork",
    "grading",
    "attendance",
    "exam",
    "teaching",
]

FEATURE_MAPPING = {
    "assignment": [
        "assignment_low_score",
        "assignment_high_score",
        "text_assignment_tfidf",
    ],
    "teamwork": [
        "teamwork_low_score",
        "teamwork_high_score",
        "text_teamwork_tfidf",
    ],
    "grading": [
        "grading_generous_score",
        "grading_strict_score",
        "text_grading_tfidf",
    ],
    "attendance": [
        "attendance_light_score",
        "attendance_strict_score",
        "text_attendance_tfidf",
    ],
    "exam": [
        "exam_light_score",
        "exam_heavy_score",
        "text_exam_tfidf",
    ],
    "teaching": [
        "text_teaching_tfidf",
    ],
}

CALIBRATED_FEATURES = [
    "assignment_low_score",
    "assignment_high_score",
    "teamwork_low_score",
    "teamwork_high_score",
    "grading_generous_score",
    "grading_strict_score",
    "attendance_light_score",
    "attendance_strict_score",
    "exam_light_score",
    "exam_heavy_score",
    "text_assignment_tfidf",
    "text_teamwork_tfidf",
    "text_grading_tfidf",
    "text_attendance_tfidf",
    "text_exam_tfidf",
    "text_teaching_tfidf",
]


# 리뷰 단위 mention vector JSON 파일을 읽는다.
def load_review_mentions(path):
    with Path(path).open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


# 강의별 feature 벡터 CSV 파일을 lecture_id 기준 딕셔너리로 읽는다.
def load_lecture_vectors(path):
    lecture_vectors = {}

    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        validate_lecture_features(reader.fieldnames or [])

        for row in reader:
            lecture_id = int(str(row["lecture_id"]).strip())
            lecture_vectors[lecture_id] = row

    return lecture_vectors


# 강의 feature 파일에 필요한 컬럼이 모두 있는지 확인한다.
def validate_lecture_features(lecture_columns):
    required_columns = {"lecture_id", *CALIBRATED_FEATURES}
    missing_columns = required_columns - set(lecture_columns)

    if missing_columns:
        raise ValueError(f"강의 feature 파일에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")


# CSV에서 읽은 feature 값을 실수로 변환한다.
def parse_feature_value(value):
    if value is None or str(value).strip() == "":
        return 0.0

    return float(str(value).strip())


# mention vector에 따라 강의 feature 값을 유지하거나 0으로 보정한다.
def create_calibrated_vector(mention_vector, lecture_row):
    calibrated_vector = {feature: 0 for feature in CALIBRATED_FEATURES}

    for category, features in FEATURE_MAPPING.items():
        if int(mention_vector.get(category, 0)) == 1:
            for feature in features:
                calibrated_vector[feature] = parse_feature_value(lecture_row[feature])

    return calibrated_vector


# 리뷰 mention 데이터와 강의 feature 데이터를 결합해 2차 벡터 데이터셋을 만든다.
def build_calibrated_dataset(review_mentions, lecture_df):
    results = []
    skipped_count = 0
    missing_lecture_ids = set()

    for review in review_mentions:
        lecture_id = int(review["lecture_id"])
        lecture_row = lecture_df.get(lecture_id)

        if lecture_row is None:
            skipped_count += 1
            missing_lecture_ids.add(lecture_id)
            continue

        mention_vector = {
            feature: int(review.get("mention_vector", {}).get(feature, 0))
            for feature in MENTION_FEATURES
        }
        results.append(
            {
                "lecture_id": lecture_id,
                "article_id": int(review["article_id"]),
                "rate": float(review["rate"]),
                "mention_vector": mention_vector,
                "calibrated_vector": create_calibrated_vector(mention_vector, lecture_row),
            }
        )

    if missing_lecture_ids:
        print(
            "warning: 강의 feature를 찾지 못한 lecture_id "
            f"{len(missing_lecture_ids)}개로 인해 리뷰 {skipped_count}개를 제외했습니다."
        )

    return results, skipped_count


# JSON과 CSV 결과 파일을 저장한다.
def save_outputs(results, json_path, csv_path):
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_columns = [
        "lecture_id",
        "article_id",
        "rate",
        *[f"mention_{feature}" for feature in MENTION_FEATURES],
        *[f"calibrated_{feature}" for feature in CALIBRATED_FEATURES],
    ]

    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)

    with csv_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        writer.writeheader()

        for result in results:
            row = {
                "lecture_id": result["lecture_id"],
                "article_id": result["article_id"],
                "rate": result["rate"],
            }
            row.update(
                {
                    f"mention_{feature}": result["mention_vector"][feature]
                    for feature in MENTION_FEATURES
                }
            )
            row.update(
                {
                    f"calibrated_{feature}": result["calibrated_vector"][feature]
                    for feature in CALIBRATED_FEATURES
                }
            )
            writer.writerow(row)


# 변환 결과의 요약 통계를 출력한다.
def print_summary(results, skipped_count):
    mention_counts = {feature: 0 for feature in MENTION_FEATURES}
    calibrated_sums = {feature: 0.0 for feature in CALIBRATED_FEATURES}
    converted_count = len(results)
    total_review_count = converted_count + skipped_count

    for result in results:
        for feature in MENTION_FEATURES:
            mention_counts[feature] += result["mention_vector"][feature]
        for feature in CALIBRATED_FEATURES:
            calibrated_sums[feature] += result["calibrated_vector"][feature]

    print(f"total_reviews: {total_review_count}")
    print(f"converted_reviews: {converted_count}")
    print(f"skipped_reviews: {skipped_count}")
    print("mention_counts:")
    for feature, count in mention_counts.items():
        print(f"- {feature}: {count}")
    print("calibrated_feature_means:")
    for feature, total in calibrated_sums.items():
        mean = total / converted_count if converted_count else 0
        print(f"- {feature}: {mean:.6f}")


# 2차 강의평가 벡터 생성 작업을 실행한다.
def main():
    review_mentions = load_review_mentions(REVIEW_MENTION_PATH)
    lecture_df = load_lecture_vectors(LECTURE_VECTOR_PATH)
    results, skipped_count = build_calibrated_dataset(review_mentions, lecture_df)
    save_outputs(results, OUTPUT_JSON_PATH, OUTPUT_CSV_PATH)
    print_summary(results, skipped_count)


if __name__ == "__main__":
    main()
