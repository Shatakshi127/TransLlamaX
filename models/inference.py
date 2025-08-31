def translate(model, tokenizer, sentences, max_len=128, beam_size=4, device="cuda:0"):
    """
    Full translation inference pipeline: Hinglish -> English
    """
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model.generate(
        **inputs,
        max_length=max_len,
        num_beams=beam_size,
        length_penalty=0.6,
        repetition_penalty=1.1,
        no_repeat_ngram_size=2
    )
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
