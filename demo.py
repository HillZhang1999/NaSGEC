from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline

valid_models = ["HillZhang/pseudo_native_bart_CGEC", ]

tokenizer = BertTokenizer.from_pretrained("HillZhang/pseudo_native_bart_CGEC")
model = BartForConditionalGeneration.from_pretrained("HillZhang/pseudo_native_bart_CGEC")
encoded_input = tokenizer(["北京是中国的都。", "他说：”我最爱的运动是打蓝球“", "我每天大约喝5次水左右。", "今天，我非常开开心。"], return_tensors="pt", padding=True, truncation=True)
if "token_type_ids" in encoded_input:
    del encoded_input["token_type_ids"]
output = model.generate(**encoded_input)
print(tokenizer.batch_decode(output, skip_special_tokens=True))

