from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import torch


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_name, output_attentions=True, trust_remote_code=True, attn_implementation="eager")
    model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True, trust_remote_code=True, attn_implementation="eager")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, model


def extract_attention_from_text(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt", max_length=1000, truncation=True)

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
    

def draw_simplicial_complex(ax, points, simplex_tree, layer_idx):
    vertices, edges, triangles = [], [], []
    for simplex, _ in simplex_tree.get_simplices():
        if len(simplex) == 1:
            vertices.append(simplex[0])
        elif len(simplex) == 2:
            edges.append(simplex)
        elif len(simplex) == 3:
            triangles.append(simplex)

    # Draw points
    shift = 0.05
    for vertex in vertices:
        x = points[vertex, 0]
        y = points[vertex, 1]
        ax.scatter(x, y, c='darkblue', s=100, zorder=5)
        ax.annotate(vertex, (x + shift, y + shift))

    # Draw lines
    for edge in edges:
        x_0, y_0 = points[edge[0]]
        x_1, y_1 = points[edge[1]]
        line = plt.Line2D([x_0, x_1], [y_0, y_1], color="blue", linewidth=1)
        ax.add_line(line)

    # Draw triangles
    for triangle in triangles:
        triangle_polygon = plt.Polygon(
            points[triangle],  # xy: (N, 2) array
            alpha=0.3,
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=1
        )
        ax.add_patch(triangle_polygon)

    ax.set_title(f"LAYER: {layer_idx}", fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    # ax.set_xticks([]), ax.set_yticks([])
