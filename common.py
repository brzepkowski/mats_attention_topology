from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# ========== MODELS ANALYSIS ==========
def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_name, output_attentions=True, trust_remote_code=True, attn_implementation="eager")
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, trust_remote_code=True, attn_implementation="eager")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def extract_attention_from_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    if outputs.attentions is None:
        raise ValueError("No attention returned by model.")
    
    logits = outputs.logits  # [batch, seq_len, vocab_size]
    next_token_logits = logits[:, -1, :]  # last position. It returns a MATRIX [batch_size x vocab_size]
    predicted_id = torch.argmax(next_token_logits, dim=-1)
    predicted_token = tokenizer.decode(predicted_id)

    return torch.stack([a for a in outputs.attentions if a is not None]).squeeze(1).detach().cpu(), predicted_token


def answer_prompt(prompt, tokenizer, model, max_new_tokens=50):
    """
    Advanced version with more control over generation parameters.
    
    Args:
        prompt (str): The input prompt/question to answer
        model_name (str): HuggingFace model name to use
        max_new_tokens (int): Maximum number of new tokens to generate
    
    Returns:
        str: Generated answer to the prompt
    """
    try:
        # # Load tokenizer and model separately for more control
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if it doesn't exist
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                # temperature=0.1,
                # do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs)
            )
        
        # Decode the response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove original prompt from response
        if response.startswith(prompt):
            answer = response[len(prompt):].strip()
        else:
            answer = response.strip()
            
        return answer
        
    except Exception as e:
        return f"Error generating response: {str(e)}"
