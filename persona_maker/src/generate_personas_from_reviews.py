import csv
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "review_calibrated_vectors.json"
ARTICLE_TEXT_PATH = PROJECT_ROOT / "data" / "csv" / "lecture_articles.csv"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "generated_personas.json"
SIMPLE_OUTPUT_PATH = PROJECT_ROOT / "outputs" / "generated_personas_simple.json"

NUM_PERSONAS = 20
TOP_K = 30
SAMPLE_N = 10
NOISE_SCALE = 0.1
RANDOM_SEED = 42

FEATURE_COLUMNS = [
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

PRESETS = {
    "low_workload": {
        "assignment_low_score": 0.9,
        "assignment_high_score": 0.1,
        "teamwork_low_score": 0.9,
        "teamwork_high_score": 0.1,
        "attendance_light_score": 0.8,
        "attendance_strict_score": 0.2,
        "exam_light_score": 0.85,
        "exam_heavy_score": 0.15,
        "grading_generous_score": 0.7,
        "grading_strict_score": 0.3,
        "text_assignment_tfidf": 0.2,
        "text_teamwork_tfidf": 0.2,
        "text_grading_tfidf": 0.2,
        "text_attendance_tfidf": 0.2,
        "text_exam_tfidf": 0.2,
        "text_teaching_tfidf": 0.3,
    },
    "learning_quality": {
        "assignment_low_score": 0.4,
        "assignment_high_score": 0.6,
        "teamwork_low_score": 0.5,
        "teamwork_high_score": 0.5,
        "attendance_light_score": 0.4,
        "attendance_strict_score": 0.6,
        "exam_light_score": 0.4,
        "exam_heavy_score": 0.6,
        "grading_generous_score": 0.4,
        "grading_strict_score": 0.6,
        "text_assignment_tfidf": 0.6,
        "text_teamwork_tfidf": 0.4,
        "text_grading_tfidf": 0.3,
        "text_attendance_tfidf": 0.3,
        "text_exam_tfidf": 0.5,
        "text_teaching_tfidf": 0.9,
    },
    "grade_focused": {
        "assignment_low_score": 0.7,
        "assignment_high_score": 0.3,
        "teamwork_low_score": 0.7,
        "teamwork_high_score": 0.3,
        "attendance_light_score": 0.6,
        "attendance_strict_score": 0.4,
        "exam_light_score": 0.7,
        "exam_heavy_score": 0.3,
        "grading_generous_score": 0.95,
        "grading_strict_score": 0.05,
        "text_assignment_tfidf": 0.2,
        "text_teamwork_tfidf": 0.2,
        "text_grading_tfidf": 0.8,
        "text_attendance_tfidf": 0.2,
        "text_exam_tfidf": 0.4,
        "text_teaching_tfidf": 0.4,
    },
    "balanced": {
        "assignment_low_score": 0.5,
        "assignment_high_score": 0.5,
        "teamwork_low_score": 0.5,
        "teamwork_high_score": 0.5,
        "attendance_light_score": 0.5,
        "attendance_strict_score": 0.5,
        "exam_light_score": 0.5,
        "exam_heavy_score": 0.5,
        "grading_generous_score": 0.5,
        "grading_strict_score": 0.5,
        "text_assignment_tfidf": 0.5,
        "text_teamwork_tfidf": 0.5,
        "text_grading_tfidf": 0.5,
        "text_attendance_tfidf": 0.5,
        "text_exam_tfidf": 0.5,
        "text_teaching_tfidf": 0.5,
    },
}

OPPOSITE_PAIRS = [
    ("assignment_low_score", "assignment_high_score"),
    ("teamwork_low_score", "teamwork_high_score"),
    ("grading_generous_score", "grading_strict_score"),
    ("attendance_light_score", "attendance_strict_score"),
    ("exam_light_score", "exam_heavy_score"),
]


# 값을 0 이상 1 이하 범위로 제한한다.
def clip_unit(value):
    return max(0.0, min(1.0, value))


# JSON 파일에서 리뷰 벡터 데이터를 읽는다.
def load_review_vectors(path):
    with Path(path).open("r", encoding="utf-8") as json_file:
        return json.load(json_file)


# article_id별 강의평가 원문 텍스트를 읽는다.
def load_article_texts(path):
    article_texts = {}

    with Path(path).open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {"article_id", "text"}
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"리뷰 원문 파일에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

        for row in reader:
            article_texts[int(row["article_id"])] = row["text"]

    return article_texts


# calibrated_vector를 고정된 feature 순서의 리스트로 변환한다.
def vector_to_list(vector):
    return [float(vector.get(feature, 0)) for feature in FEATURE_COLUMNS]


# 벡터가 모두 0인지 확인한다.
def is_all_zero(vector):
    return all(float(vector.get(feature, 0)) == 0 for feature in FEATURE_COLUMNS)


# 코사인 유사도를 계산한다.
def cosine_similarity(left_vector, right_vector):
    dot_product = sum(left * right for left, right in zip(left_vector, right_vector))
    left_norm = math.sqrt(sum(value * value for value in left_vector))
    right_norm = math.sqrt(sum(value * value for value in right_vector))

    if left_norm == 0 or right_norm == 0:
        return 0.0

    return dot_product / (left_norm * right_norm)


# 반대 의미 feature 쌍의 합이 1이 되도록 보정한다.
def normalize_opposite_pairs(preference_vector):
    for low_feature, high_feature in OPPOSITE_PAIRS:
        low_value = preference_vector[low_feature]
        high_value = preference_vector[high_feature]
        total = low_value + high_value

        if total == 0:
            preference_vector[low_feature] = 0.5
            preference_vector[high_feature] = 0.5
        else:
            preference_vector[low_feature] = low_value / total
            preference_vector[high_feature] = high_value / total

    return preference_vector


# preset에 랜덤 노이즈를 더해 초기 선호도 벡터를 생성한다.
def create_random_preference_vector(preset, noise_scale):
    preference_vector = {}

    for feature in FEATURE_COLUMNS:
        base_value = float(preset.get(feature, 0))
        noisy_value = base_value + random.uniform(-noise_scale, noise_scale)
        preference_vector[feature] = clip_unit(noisy_value)

    return normalize_opposite_pairs(preference_vector)


# 유사도 계산에 사용할 수 있는 리뷰와 제외 개수를 분리한다.
def filter_valid_reviews(review_vectors):
    valid_reviews = []
    all_zero_count = 0

    for review in review_vectors:
        if is_all_zero(review.get("calibrated_vector", {})):
            all_zero_count += 1
            continue
        valid_reviews.append(review)

    return valid_reviews, all_zero_count


# 선호도 벡터와 가까운 Top-K 리뷰 후보를 찾는다.
def find_top_k_reviews(preference_vector, valid_reviews, top_k):
    preference_values = vector_to_list(preference_vector)
    scored_reviews = []

    for review in valid_reviews:
        review_values = vector_to_list(review["calibrated_vector"])
        similarity = cosine_similarity(preference_values, review_values)
        scored_reviews.append((similarity, review))

    scored_reviews.sort(key=lambda item: item[0], reverse=True)
    return scored_reviews[:top_k]


# 선택된 리뷰들의 calibrated_vector 평균을 계산한다.
def aggregate_review_vector(selected_reviews):
    if not selected_reviews:
        return {feature: 0 for feature in FEATURE_COLUMNS}

    aggregated_vector = {}
    for feature in FEATURE_COLUMNS:
        total = sum(
            float(review["calibrated_vector"].get(feature, 0))
            for review in selected_reviews
        )
        aggregated_vector[feature] = total / len(selected_reviews)

    return aggregated_vector


# persona에 포함할 리뷰 객체를 생성한다.
def build_selected_review(scored_review, article_texts):
    similarity, review = scored_review
    article_id = int(review["article_id"])

    return {
        "lecture_id": review["lecture_id"],
        "article_id": article_id,
        "text": article_texts.get(article_id, ""),
        "rate": review["rate"],
        "similarity": round(similarity, 6),
        "mention_vector": review["mention_vector"],
        "calibrated_vector": review["calibrated_vector"],
    }


# 하나의 preset 기반 persona를 생성한다.
def create_persona(persona_id, preset_name, valid_reviews, article_texts):
    preference_vector = create_random_preference_vector(
        PRESETS[preset_name],
        NOISE_SCALE,
    )
    top_k_reviews = find_top_k_reviews(preference_vector, valid_reviews, TOP_K)
    sample_size = min(SAMPLE_N, len(top_k_reviews))
    sampled_scored_reviews = random.sample(top_k_reviews, sample_size)
    selected_reviews = [
        build_selected_review(scored_review, article_texts)
        for scored_review in sampled_scored_reviews
    ]

    return {
        "persona_id": persona_id,
        "preset_name": preset_name,
        "initial_preference_vector": preference_vector,
        "top_k": TOP_K,
        "sample_n": sample_size,
        "selected_reviews": selected_reviews,
        "aggregated_review_vector": aggregate_review_vector(selected_reviews),
    }


# 여러 개의 persona 데이터를 생성한다.
def generate_personas(valid_reviews, article_texts):
    personas = []
    preset_names = list(PRESETS)

    for persona_id in range(1, NUM_PERSONAS + 1):
        preset_name = random.choice(preset_names)
        personas.append(
            create_persona(persona_id, preset_name, valid_reviews, article_texts)
        )

    return personas


# 생성된 persona 데이터를 JSON 파일로 저장한다.
def save_personas(path, personas):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(personas, json_file, ensure_ascii=False, indent=2)


# 간소화된 persona 데이터만 추출한다.
def simplify_personas(personas):
    simple_personas = []

    for persona in personas:
        simple_personas.append(
            {
                "persona_id": persona["persona_id"],
                "preset_name": persona["preset_name"],
                "initial_preference_vector": persona["initial_preference_vector"],
                "selected_reviews": [
                    {"text": review["text"]}
                    for review in persona["selected_reviews"]
                ],
            }
        )

    return simple_personas


# persona별 평균 similarity를 계산한다.
def calculate_persona_average_similarity(persona):
    selected_reviews = persona["selected_reviews"]
    if not selected_reviews:
        return 0.0

    total = sum(review["similarity"] for review in selected_reviews)
    return total / len(selected_reviews)


# 생성 결과 요약 통계를 출력한다.
def print_summary(total_reviews, used_reviews, all_zero_count, personas):
    preset_counts = Counter(persona["preset_name"] for persona in personas)

    print(f"total_reviews: {total_reviews}")
    print(f"used_reviews: {used_reviews}")
    print(f"excluded_all_zero_reviews: {all_zero_count}")
    print(f"generated_personas: {len(personas)}")
    print("preset_counts:")
    for preset_name in PRESETS:
        print(f"- {preset_name}: {preset_counts[preset_name]}")
    print("persona_average_similarity:")
    for persona in personas:
        average_similarity = calculate_persona_average_similarity(persona)
        print(f"- persona_{persona['persona_id']}: {average_similarity:.6f}")


# 실행 인자를 파싱한다.
def parse_args():
    parser = argparse.ArgumentParser(
        description="강의평가 기반 preset 랜덤 페르소나 데이터를 생성합니다."
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="간소화된 generated_personas_simple.json 파일을 저장합니다.",
    )
    return parser.parse_args()


# 페르소나 생성 작업을 실행한다.
def main():
    args = parse_args()
    random.seed(RANDOM_SEED)

    review_vectors = load_review_vectors(INPUT_PATH)
    article_texts = load_article_texts(ARTICLE_TEXT_PATH)
    valid_reviews, all_zero_count = filter_valid_reviews(review_vectors)
    personas = generate_personas(valid_reviews, article_texts)
    output_path = SIMPLE_OUTPUT_PATH if args.simple else OUTPUT_PATH
    output_personas = simplify_personas(personas) if args.simple else personas
    save_personas(output_path, output_personas)
    print_summary(
        total_reviews=len(review_vectors),
        used_reviews=len(valid_reviews),
        all_zero_count=all_zero_count,
        personas=personas,
    )


if __name__ == "__main__":
    main()
