import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "csv" / "lecture_articles.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "review_mention_vectors.json"

KEYWORDS = {
    "assignment": ["과제", "숙제", "레포트", "보고서", "프로젝트", "과제량"],
    "teamwork": ["팀플", "조별", "조모임", "팀프로젝트", "조별과제"],
    "grading": ["학점", "성적", "평가", "A+", "에이쁠", "비율", "곡선"],
    "attendance": ["출석", "결석", "지각", "출결", "출첵", "전자출결"],
    "exam": ["시험", "중간", "기말", "퀴즈", "고사", "족보"],
    "teaching": ["강의력", "설명", "교수님", "수업", "이해", "전달력", "피드백"],
}


# 리뷰 텍스트에서 요소별 언급 여부를 판단한다.
def classify_mention_vector(text):
    normalized_text = str(text or "").lower()
    mention_vector = {}

    for feature, keywords in KEYWORDS.items():
        mention_vector[feature] = int(
            any(keyword.lower() in normalized_text for keyword in keywords)
        )

    return mention_vector


# 문자열 값을 정수로 변환한다.
def parse_int(value):
    return int(str(value).strip())


# 문자열 값을 실수로 변환한다.
def parse_float(value):
    return float(str(value).strip())


# CSV 한 행을 JSON으로 저장할 리뷰 객체로 변환한다.
def build_review_object(row):
    return {
        "lecture_id": parse_int(row["lecture_id"]),
        "article_id": parse_int(row["article_id"]),
        "rate": parse_float(row["rate"]),
        "mention_vector": classify_mention_vector(row["text"]),
    }


# mention vector별 언급 횟수 통계를 계산한다.
def summarize_mentions(review_objects):
    summary = {feature: 0 for feature in KEYWORDS}

    for review_object in review_objects:
        for feature, value in review_object["mention_vector"].items():
            summary[feature] += value

    return summary


# CSV 파일을 읽어 리뷰 단위 JSON 객체 목록을 생성한다.
def load_review_objects(input_path):
    required_columns = {"lecture_id", "article_id", "rate", "text"}
    review_objects = []

    with input_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {sorted(missing_columns)}")

        for row in reader:
            review_objects.append(build_review_object(row))

    return review_objects


# 리뷰 객체 목록을 JSON 파일로 저장한다.
def save_review_objects(output_path, review_objects):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(review_objects, json_file, ensure_ascii=False, indent=2)


# 실행 인자를 파싱한다.
def parse_args():
    parser = argparse.ArgumentParser(
        description="강의평가 CSV를 리뷰 단위 mention vector JSON으로 변환합니다."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT_PATH),
        help="입력 CSV 파일 경로",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="출력 JSON 파일 경로",
    )
    return parser.parse_args()


# 변환 작업을 실행하고 요약 통계를 출력한다.
def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    review_objects = load_review_objects(input_path)
    save_review_objects(output_path, review_objects)
    summary = summarize_mentions(review_objects)

    print(f"saved: {output_path}")
    print(f"total_reviews: {len(review_objects)}")
    print("mention_summary:")
    for feature, count in summary.items():
        print(f"- {feature}: {count}")


if __name__ == "__main__":
    main()
