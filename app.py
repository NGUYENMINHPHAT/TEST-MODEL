import os
import openai
import pandas as pd
from flask import Flask, request, render_template, jsonify
import difflib
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

openai.api_key = os.getenv("OPENAI_KEY")


def query_openai_model(prompt, model):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return str(e)


def similarity_score(expected, returned):
    """
    Tính toán độ tương đồng giữa hai chuỗi sử dụng SequenceMatcher của difflib
    """
    return difflib.SequenceMatcher(None, expected, returned).ratio()


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        model = request.form["model"]
        file = request.files["file"]

        if file and model:
            data = pd.read_excel(file)
            if "prompt" in data.columns and "completion" in data.columns:
                total = len(data)
                correct = 0
                wrong_answers = []
                threshold = 0.0

                for idx, row in data.iterrows():
                    prompt = row["prompt"]
                    expected_completion = row["completion"]
                    if pd.isna(expected_completion):
                        continue

                    if not isinstance(expected_completion, str):
                        expected_completion = str(expected_completion)

                    # Xử lý định dạng chuỗi
                    expected_completion = " ".join(expected_completion.split())
                    returned_answer = query_openai_model(prompt, model)
                    returned_answer = " ".join(returned_answer.split())

                    # So sánh dựa trên tỷ lệ tương đồng
                    similarity = similarity_score(expected_completion, returned_answer)

                    if similarity >= threshold:
                        correct += 1
                    else:
                        wrong_answers.append(
                            {
                                "STT": idx + 1,
                                "Prompt": prompt,
                                "Expected": expected_completion,
                                "Returned": returned_answer,
                                "Similarity": f"{similarity * 100:.2f}%",  # Hiển thị tỉ lệ % giống nhau
                            }
                        )

                wrong_count = len(wrong_answers)
                correct_count = correct
                return render_template(
                    "result.html",
                    correct=correct_count,
                    wrong_count=wrong_count,
                    wrong_answers=wrong_answers,
                )
            else:
                return (
                    "File không đúng định dạng. Phải có cột 'prompt' và 'completion'."
                )
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
