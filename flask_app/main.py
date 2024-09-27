#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, request, render_template
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("./saved_model/", local_files_only=True)
model = AutoModelForQuestionAnswering.from_pretrained("./saved_model/", local_files_only=True)
model.to(device)

max_length = 128

app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def hello_world():
    if request.method == "POST":
        context = request.form.get("context").strip()
        question = request.form.get("question").strip()
        inputs = tokenizer(
            question, 
            context,
            max_length=max_length,
            truncation="only_second",
            padding="max_length",
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        answer_start_index = outputs.start_logits.argmax()
        answer_end_index = outputs.end_logits.argmax()
        predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
        return render_template("index.html", context=context, question=question, answer=tokenizer.decode(predict_answer_tokens))
    elif request.method == "GET":
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")
